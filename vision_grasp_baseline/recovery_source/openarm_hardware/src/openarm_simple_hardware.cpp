// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "openarm_hardware/openarm_simple_hardware.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/socket.h>
#include <fcntl.h>
#include "openarm_hardware/recovery_feedback.hpp"

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace openarm_hardware {

OpenArmHW::OpenArmHW() = default;

OpenArmHW::~OpenArmHW() { shutdown_passive_control(); }

bool OpenArmHW::parse_config(const hardware_interface::HardwareInfo& info) {
  const auto hold = info.hardware_parameters.find("startup_hold_current");
  startup_hold_current_ = hold != info.hardware_parameters.end() && hold->second == "true";
  // Parse CAN interface (default: can0)
  auto it = info.hardware_parameters.find("can_interface");
  can_interface_ = (it != info.hardware_parameters.end()) ? it->second : "can0";

  // Parse arm prefix (default: empty for single arm, "left_" or "right_" for
  // bimanual)
  it = info.hardware_parameters.find("arm_prefix");
  arm_prefix_ = (it != info.hardware_parameters.end()) ? it->second : "";

  // Parse gripper enable (default: true for V10)
  it = info.hardware_parameters.find("hand");
  if (it == info.hardware_parameters.end()) {
    hand_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    hand_ = (value == "true");
  }

  // Parse CAN-FD enable (default: true for V10)
  it = info.hardware_parameters.find("can_fd");
  if (it == info.hardware_parameters.end()) {
    can_fd_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    can_fd_ = (value == "true");
  }

  // Parse control gains
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    it = info.hardware_parameters.find("kp" + std::to_string(i));
    if (it != info.hardware_parameters.end()) {
      kp_[i - 1] = std::stod(it->second);
    }
    it = info.hardware_parameters.find("kd" + std::to_string(i));
    if (it != info.hardware_parameters.end()) {
      kd_[i - 1] = std::stod(it->second);
    }
  }
  // Parse ee_type (default: parallel_link for v10)
  it = info.hardware_parameters.find("ee_type");
  ee_type_ =
      (it != info.hardware_parameters.end()) ? it->second : "parallel_link";

  // Parse gravity compensation parameters (default: disabled)
  it = info.hardware_parameters.find("gravity_compensation");
  if (it != info.hardware_parameters.end()) {
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    gravity_compensation_ = (value == "true");
  }
  it = info.hardware_parameters.find("gravity_scale");
  if (it != info.hardware_parameters.end()) {
    gravity_scale_ = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gravity_urdf_path");
  gravity_urdf_path_ = (it != info.hardware_parameters.end()) ? it->second : "";
  it = info.hardware_parameters.find("gravity_base_link");
  gravity_base_link_ = (it != info.hardware_parameters.end()) ? it->second : "";
  it = info.hardware_parameters.find("gravity_tip_link");
  gravity_tip_link_ = (it != info.hardware_parameters.end()) ? it->second : "";
  it = info.hardware_parameters.find("passive_heartbeat_timeout");
  if (it != info.hardware_parameters.end()) {
    passive_heartbeat_timeout_ = std::stod(it->second);
  }

  if (hand_) {
    it = info.hardware_parameters.find("kp_hand");
    if (it != info.hardware_parameters.end()) {
      gripper_kp_ = std::stod(it->second);
    }
    it = info.hardware_parameters.find("kd_hand");
    if (it != info.hardware_parameters.end()) {
      gripper_kd_ = std::stod(it->second);
    }
  }

  // --- Gripper contact detection parameters (all optional, safe defaults) ---
  // The contact detector is DISABLED by default so existing behavior is
  // unchanged unless explicitly enabled through the URDF/xacro params.
  auto contact_mode_from_string = [](const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "enforce") {
      return GripperContactMode::kEnforce;
    }
    if (v == "monitor") {
      return GripperContactMode::kMonitor;
    }
    return GripperContactMode::kDisabled;
  };
  it = info.hardware_parameters.find("gripper_contact_mode");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.mode = contact_mode_from_string(it->second);
  }
  it = info.hardware_parameters.find("gripper_torque_threshold");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.torque_threshold = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_torque_release_threshold");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.torque_release_threshold = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_hard_torque_limit");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.hard_torque_limit = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_torque_filter_alpha");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.filter_alpha = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_contact_confirm_cycles");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.confirm_cycles = std::stoi(it->second);
  }
  it = info.hardware_parameters.find("gripper_hold_kp_scale");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.hold_kp_scale = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_hold_kd_scale");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.hold_kd_scale = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_hold_position_offset");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.hold_position_offset = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_close_timeout");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.max_close_duration_sec = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_position_reached_tolerance");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.position_reached_tolerance = std::stod(it->second);
  }
  it = info.hardware_parameters.find("gripper_contact_log_rate");
  if (it != info.hardware_parameters.end()) {
    gripper_log_rate_hz_ = std::stod(it->second);
  }
  // Closing direction follows the ee_type convention:
  //   pinch_gripper (v2.0 revolute): joint 0 = closed. Right hand opens
  //     negative, closing = joint increasing toward 0 -> +1; the left hand is
  //     the mirror image (opens positive), closing = joint decreasing -> -1.
  //   parallel_link (v1.x prismatic): 0.044 open -> 0 closed -> -1
  if (ee_type_ == "parallel_link") {
    gripper_contact_config_.closing_direction = -1.0;
  } else {
    gripper_contact_config_.closing_direction =
        (arm_prefix_ == "left_") ? -1.0 : 1.0;
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "Configuration: CAN=%s, arm_prefix=%s, hand=%s, can_fd=%s, "
              "gravity_comp=%s, gravity_scale=%.3f, "
              "gripper_contact_mode=%s, torque_threshold=%.3f Nm, "
              "hard_torque_limit=%.3f Nm",
              can_interface_.c_str(), arm_prefix_.c_str(),
              hand_ ? "enabled" : "disabled", can_fd_ ? "enabled" : "disabled",
              gravity_compensation_ ? "enabled" : "disabled", gravity_scale_,
              GripperContactDetector::mode_name(gripper_contact_config_.mode),
              gripper_contact_config_.torque_threshold,
              gripper_contact_config_.hard_torque_limit);
  return true;
}

void OpenArmHW::generate_joint_names() {
  joint_names_.clear();
  // TODO: read from urdf properly and sort in the future.
  // Currently, the joint names are hardcoded for order consistency to align
  // with hardware. Generate arm joint names: openarm_{arm_prefix}joint{N}
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    std::string joint_name =
        "openarm_" + arm_prefix_ + "joint" + std::to_string(i);
    joint_names_.push_back(joint_name);
  }

  // Generate gripper joint name if enabled
  if (hand_) {
    std::string gripper_joint_name = "openarm_" + arm_prefix_ + "finger_joint1";
    joint_names_.push_back(gripper_joint_name);
    RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Added gripper joint: %s",
                gripper_joint_name.c_str());
  } else {
    RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
                "Gripper joint NOT added because hand_=false");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "Generated %zu joint names for arm prefix '%s'",
              joint_names_.size(), arm_prefix_.c_str());
}

hardware_interface::CallbackReturn OpenArmHW::on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) {
  if (hardware_interface::SystemInterface::on_init(params) !=
      CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  // Parse configuration
  if (!parse_config(params.hardware_info)) {
    return CallbackReturn::ERROR;
  }

  // Generate joint names based on arm prefix
  generate_joint_names();

  // Validate joint count (7 arm joints + optional gripper)
  size_t expected_joints = ARM_DOF + (hand_ ? 1 : 0);
  if (joint_names_.size() != expected_joints) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"),
                 "Generated %zu joint names, expected %zu", joint_names_.size(),
                 expected_joints);
    return CallbackReturn::ERROR;
  }

  // Lock BEFORE constructing the SDK: initialization may itself transmit CAN.
  // Refusal cannot enable, disable, zero, refresh, or otherwise touch a motor.
  try {
    can_ownership_ = std::make_unique<CanOwnership>(
        can_interface_, "ros2_control:" + arm_prefix_);
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"), "%s", error.what());
    return CallbackReturn::ERROR;
  }

  // Initialize OpenArm with configurable CAN-FD setting
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "Initializing OpenArm on %s with CAN-FD %s...",
              can_interface_.c_str(), can_fd_ ? "enabled" : "disabled");
  openarm_ =
      std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

  // Initialize arm motors with V10 defaults
  openarm_->init_arm_motors(DEFAULT_MOTOR_TYPES, DEFAULT_SEND_CAN_IDS,
                            DEFAULT_RECV_CAN_IDS);

  // Initialize gripper if enabled
  if (hand_) {
    RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Initializing gripper...");
    openarm_->init_gripper_motor(DEFAULT_GRIPPER_MOTOR_TYPE,
                                 DEFAULT_GRIPPER_SEND_CAN_ID,
                                 DEFAULT_GRIPPER_RECV_CAN_ID);
  }

  // Initialize state and command vectors based on generated joint count
  const size_t total_joints = joint_names_.size();
  pos_commands_.resize(total_joints, 0.0);
  vel_commands_.resize(total_joints, 0.0);
  tau_commands_.resize(total_joints, 0.0);
  pos_states_.resize(total_joints, 0.0);
  vel_states_.resize(total_joints, 0.0);
  tau_states_.resize(total_joints, 0.0);

  // Gripper contact detector (shared logic with the simulation plugin).
  // configure() arms it; in disabled mode it stays inert so the legacy
  // behavior is fully preserved.
  gripper_contact_ = std::make_unique<GripperContactDetector>();
  gripper_contact_->configure(gripper_contact_config_);
  setup_gripper_diagnostics();

  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "OpenArm V10 Simple HW initialized successfully");

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArmHW::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Initialize KDL gravity compensation if enabled
  if (gravity_compensation_) {
    if (gravity_base_link_.empty()) {
      gravity_base_link_ = "openarm_" + arm_prefix_ + "base_link";
    }
    if (gravity_tip_link_.empty()) {
      gravity_tip_link_ = "openarm_" + arm_prefix_ + "ee_base_link";
    }
    dynamics_ = std::make_unique<Dynamics>(gravity_urdf_path_,
                                           gravity_base_link_,
                                           gravity_tip_link_);
    if (dynamics_->Init() && dynamics_->JointCount() == ARM_DOF) {
      gravity_ff_.resize(dynamics_->JointCount());
      gravity_ready_.store(true);
      RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
                  "Gravity compensation enabled: chain %s -> %s, "
                  "urdf=%s, scale=%.3f",
                  gravity_base_link_.c_str(), gravity_tip_link_.c_str(),
                  gravity_urdf_path_.c_str(), gravity_scale_);
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"),
                   "Gravity compensation initialization FAILED "
                   "(chain must have %zu joints); disabled",
                   ARM_DOF);
      dynamics_.reset();
      gravity_ff_.clear();
      gravity_ready_.store(false);
    }
  } else {
    gravity_ready_.store(false);
  }

  // Set callback mode to ignore during configuration
  openarm_->refresh_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  // Seed the gripper command with the current feedback position: the command
  // interface defaults to 0.0 (= closed), so without this the gripper would
  // slam shut (and could trip the contact detector / hard limit) during
  // bring-up before the gripper controller is loaded.
  if (hand_ && joint_names_.size() > ARM_DOF) {
    pos_commands_[ARM_DOF] = pos_states_[ARM_DOF];
  }

  setup_passive_control();

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
OpenArmHW::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
OpenArmHW::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  // TODO: consider exposing only needed interfaces to avoid undefined behavior.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION,
        &pos_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY,
        &vel_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn OpenArmHW::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (startup_hold_current_) return activate_hold_current();
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Activating OpenArm V10...");
  openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  openarm_->enable_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  // Return to zero position
  return_to_zero();

  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "OpenArm V10 activated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArmHW::activate_hold_current() {
  // The existing on_activate/default return_to_zero path remains untouched.
  // This opt-in path never calls return_to_zero or set_zero.
  int fd = -1;
  bool enabled = false;
  try {
    if (!hand_ || ee_type_ != "pinch_gripper" || !gravity_ready_.load())
      throw std::runtime_error("Recovery startup requires V2 gripper and working gravity compensation");
    fd = ::socket(PF_CAN, SOCK_RAW | SOCK_CLOEXEC | SOCK_NONBLOCK, CAN_RAW);
    if (fd < 0) throw std::runtime_error("Cannot open feedback observer");
    int frames = 1;
    if (::setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &frames, sizeof(frames)))
      throw std::runtime_error("Cannot enable FD feedback");
    sockaddr_can address{}; address.can_family = AF_CAN;
    address.can_ifindex = ::if_nametoindex(can_interface_.c_str());
    if (!address.can_ifindex || ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)))
      throw std::runtime_error("Cannot bind feedback observer");
    openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    auto sample = [&]() {
      RecoveryFeedback result;
      canfd_frame frame{};
      while (::read(fd, &frame, sizeof(frame)) > 0) {} // discard preceding phase
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
      while (std::chrono::steady_clock::now() < deadline) {
        for (unsigned i = 0; i < 8; ++i) {
          if (i < ARM_DOF) openarm_->get_arm().refresh_one(i);
          else openarm_->get_gripper().refresh_one(0);
          std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        openarm_->recv_all();
        ssize_t n;
        while ((n = ::read(fd, &frame, sizeof(frame))) > 0)
          if (n == CAN_MTU || n == CANFD_MTU) result.accept(frame);
        if (std::all_of(result.seen.begin(), result.seen.end(), [](bool v) { return v; })) return result;
      }
      return result;
    };
    const auto before = sample();
    before.validate(0); // no enables on absent feedback, moving joints, or faults
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      const double q = i < ARM_DOF ? before.position[i] : motor_radians_to_joint(before.position[i]);
      pos_states_[i] = pos_commands_[i] = q;
      vel_states_[i] = vel_commands_[i] = tau_commands_[i] = 0.;
    }
    // Preload the SAME measured targets while still disabled. No zero target.
    write(rclcpp::Time(0), rclcpp::Duration::from_seconds(.01));
    openarm_->enable_all(); enabled = true;
    // Keep target current throughout activation, including the first enabled write.
    write(rclcpp::Time(0), rclcpp::Duration::from_seconds(.01));
    const auto after = sample();
    after.validate(1);
    for (size_t i = 0; i < ARM_DOF; ++i)
      if (std::abs(after.position[i] - before.position[i]) > .05)
        throw std::runtime_error("Unexpected movement while enabling measured-position hold");
    for (size_t i = 0; i < ARM_DOF; ++i)
      RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "RECOVERY_HOLD %s J%zu measured=%.6f first_target=%.6f enabled=%.6f",
                  arm_prefix_.c_str(), i+1, before.position[i], pos_commands_[i], after.position[i]);
    ::close(fd);
    RCLCPP_WARN(rclcpp::get_logger("OpenArmHW"), "Recovery startup holding measured pose on %s; zero motion requires separately planned H2 trajectory", can_interface_.c_str());
    return CallbackReturn::SUCCESS;
  } catch (const std::exception& exc) {
    if (enabled) openarm_->disable_all();
    if (fd >= 0) ::close(fd);
    RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"), "Recovery hold startup failed: %s", exc.what());
    return CallbackReturn::ERROR;
  }
}

hardware_interface::CallbackReturn OpenArmHW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Deactivating OpenArm V10...");
  set_passive_requested(false);

  // Disable all motors (like full_arm.cpp exit)
  for (int i = 0; i < 3; ++i) {
    openarm_->disable_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    openarm_->recv_all();
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "OpenArm V10 deactivated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type OpenArmHW::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // Receive all motor states
  openarm_->refresh_all();
  openarm_->recv_all();

  // Read arm joint states
  const auto& arm_motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    pos_states_[i] = arm_motors[i].get_position();
    vel_states_[i] = arm_motors[i].get_velocity();
    tau_states_[i] = arm_motors[i].get_torque();
  }

  // Read gripper state if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    const auto& gripper_motors = openarm_->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      double motor_pos = gripper_motors[0].get_position();
      double motor_vel = gripper_motors[0].get_velocity();
      pos_states_[ARM_DOF] = motor_radians_to_joint(motor_pos);
      if (ee_type_ == "pinch_gripper") {
        // revolute 1:1 mapping
        vel_states_[ARM_DOF] = motor_vel;
      } else {
        // parallel_link prismatic: scale rad/s to m/s (same linear factor)
        vel_states_[ARM_DOF] =
            motor_vel * (GRIPPER_JOINT_0_POSITION / GRIPPER_MOTOR_1_RADIANS);
      }
      // Motor-side torque (Nm) is exposed as-is; it is the raw contact signal
      // for GripperContactDetector. It may be NaN until a valid CAN frame
      // arrives, which the detector treats as sensor_invalid.
      tau_states_[ARM_DOF] = gripper_motors[0].get_torque();
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OpenArmHW::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  bool passive = passive_requested_.load();
  if (passive) {
    const int64_t heartbeat_ns = passive_heartbeat_ns_.load();
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    const double age = static_cast<double>(now_ns - heartbeat_ns) * 1e-9;
    if (heartbeat_ns == 0 || age > passive_heartbeat_timeout_) {
      passive_requested_.store(false);
      passive = false;
      RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"),
                   "Passive heartbeat timed out after %.3f s on arm '%s'; "
                   "requesting safe measured-position rigid hold",
                   age, arm_prefix_.c_str());
    }
  }
  // P0 invariant: on every passive -> rigid edge (including heartbeat
  // timeout), overwrite every arm command interface from the latest measured
  // state in this same hardware write cycle, before any non-zero arm kp is used.
  // This makes an old JTC/HOME/setpoint physically incapable of becoming the
  // first rigid target.
  if (!passive && passive_applied_) {
    const bool measured_valid = std::all_of(
        pos_states_.begin(), pos_states_.begin() + ARM_DOF,
        [](double value) { return std::isfinite(value); });
    if (!measured_valid) {
      passive = true;
      passive_requested_.store(true);
      RCLCPP_ERROR(rclcpp::get_logger("OpenArmHW"),
                   "Refusing rigid gains on arm '%s': measured J1-J7 are not "
                   "finite; remaining hardware-passive",
                   arm_prefix_.c_str());
    } else {
      double maximum_stale_error = 0.0;
      for (size_t i = 0; i < ARM_DOF; ++i) {
        maximum_stale_error =
            std::max(maximum_stale_error,
                     std::abs(pos_commands_[i] - pos_states_[i]));
        pos_commands_[i] = pos_states_[i];
        vel_commands_[i] = 0.0;
        tau_commands_[i] = 0.0;
      }
      RCLCPP_WARN(rclcpp::get_logger("OpenArmHW"),
                  "Rigid target synchronized from fresh measured J1-J7 on "
                  "arm '%s' before kp restore (discarded stale max error="
                  "%.6f rad)",
                  arm_prefix_.c_str(), maximum_stale_error);
    }
  }
  if (passive != passive_applied_) {
    passive_applied_ = passive;
    publish_passive_state(passive);
    RCLCPP_WARN(rclcpp::get_logger("OpenArmHW"),
                "Hardware passive mode %s for arm '%s' (J1-J7 kp=%s, J8 remains "
                "electrically controlled, arm kd+gravity retained)",
                passive ? "ENABLED" : "DISABLED", arm_prefix_.c_str(),
                passive ? "0" : "configured");
  }

  // Compute gravity feedforward torque G(q) from current joint states
  if (dynamics_ && gravity_ff_.size() == ARM_DOF) {
    dynamics_->GetGravity(pos_states_.data(), gravity_ff_.data());
  }

  // Control arm motors with MIT control
  std::vector<openarm::damiao_motor::MITParam> arm_params;
  for (size_t i = 0; i < ARM_DOF; ++i) {
    double ff_tau = tau_commands_[i];
    if (dynamics_ && gravity_ff_.size() == ARM_DOF) {
      ff_tau += gravity_scale_ * gravity_ff_[i];
    }
    const double effective_kp = passive ? 0.0 : kp_[i];
    const double effective_position = passive ? pos_states_[i] : pos_commands_[i];
    const double effective_velocity = passive ? 0.0 : vel_commands_[i];
    arm_params.push_back({effective_kp, kd_[i], effective_position,
                          effective_velocity, ff_tau});
  }
  openarm_->get_arm().mit_control_all(arm_params);
  // Control gripper if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    const double dt = period.seconds();

    // Feed the raw motor torque back into the contact state machine. In
    // enforce mode the detector may freeze the effective close target and
    // scale down the gains once a reliable contact is latched.
    update_gripper_contact(tau_states_[ARM_DOF], pos_states_[ARM_DOF],
                           vel_states_[ARM_DOF], pos_commands_[ARM_DOF], dt);

    // Arm passive mode does not disable J8.  Keep the gripper controller and
    // configured gains live so LT+LB+L3 can open/close it during teaching.
    double gripper_command = pos_commands_[ARM_DOF];
    double gkp = gripper_kp_;
    double gkd = gripper_kd_;
    if (gripper_contact_) {
      gripper_command = gripper_contact_->effective_command(gripper_command);
      gkp *= gripper_contact_->kp_scale();
      gkd *= gripper_contact_->kd_scale();
    }

    double motor_command = joint_to_motor_radians(gripper_command);
    openarm_->get_gripper().mit_control_all(
        {{gkp, gkd, motor_command, 0.0, 0.0}});

    publish_gripper_diagnostics();
    log_gripper_contact_throttled();
  }
  openarm_->recv_all(100);
  return hardware_interface::return_type::OK;
}

void OpenArmHW::setup_passive_control() {
  if (control_node_) {
    return;
  }
  std::string side = arm_prefix_;
  while (!side.empty() && side.back() == '_') {
    side.pop_back();
  }
  const std::string node_name =
      side.empty() ? "openarm_hardware" : "openarm_" + side + "_hardware";
  control_node_ = std::make_shared<rclcpp::Node>(node_name);
  passive_state_pub_ = control_node_->create_publisher<std_msgs::msg::Bool>(
      "~/passive_state", rclcpp::QoS(1).reliable().transient_local());
  passive_service_ = control_node_->create_service<std_srvs::srv::SetBool>(
      "~/set_passive",
      [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
             std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        if (request->data && !gravity_ready_.load()) {
          response->success = false;
          response->message =
              "refused: gravity compensation is not initialized";
          return;
        }
        set_passive_requested(request->data);
        response->success = true;
        response->message = request->data
                                ? "hardware passive enabled: J1-J7 kp=0; J8 remains controlled"
                                : "hardware passive disabled: rigid gains restored";
      });
  passive_heartbeat_sub_ = control_node_->create_subscription<std_msgs::msg::Bool>(
      "~/passive_heartbeat", rclcpp::QoS(10).reliable(),
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data) {
          set_passive_requested(false);
          return;
        }
        passive_heartbeat_ns_.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
      });
  control_executor_ =
      std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  control_executor_->add_node(control_node_);
  control_thread_ = std::thread([executor = control_executor_]() {
    executor->spin();
  });
  publish_passive_state(false);
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "Hardware passive control ready: /%s/set_passive, heartbeat "
              "timeout=%.3f s",
              node_name.c_str(), passive_heartbeat_timeout_);
}

void OpenArmHW::shutdown_passive_control() {
  passive_requested_.store(false);
  if (control_executor_) {
    control_executor_->cancel();
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  if (control_executor_ && control_node_) {
    control_executor_->remove_node(control_node_);
  }
  passive_heartbeat_sub_.reset();
  passive_service_.reset();
  passive_state_pub_.reset();
  control_node_.reset();
  control_executor_.reset();
}

void OpenArmHW::set_passive_requested(bool requested) {
  if (requested) {
    passive_heartbeat_ns_.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  } else {
    passive_heartbeat_ns_.store(0);
  }
  passive_requested_.store(requested);
}

void OpenArmHW::publish_passive_state(bool enabled) {
  if (!passive_state_pub_) {
    return;
  }
  std_msgs::msg::Bool msg;
  msg.data = enabled;
  passive_state_pub_->publish(msg);
}

void OpenArmHW::setup_gripper_diagnostics() {
  if (gripper_contact_config_.mode == GripperContactMode::kDisabled) {
    return;  // nothing to set up, legacy behavior untouched
  }
  // A lightweight side node so the hardware plugin can publish diagnostics.
  // The suffix keeps the node name unique across bimanual instances.
  diag_node_ =
      std::make_shared<rclcpp::Node>("openarm_hw_diagnostics_" + arm_prefix_);
  const std::string base = arm_prefix_ + "gripper";
  contact_state_pub_ = diag_node_->create_publisher<std_msgs::msg::Int32>(
      base + "/contact_state", rclcpp::SensorDataQoS());
  motor_torque_pub_ = diag_node_->create_publisher<std_msgs::msg::Float64>(
      base + "/motor_torque", rclcpp::SensorDataQoS());
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"),
              "Gripper contact diagnostics on %s/contact_state, "
              "%s/motor_torque (mode=%s)",
              base.c_str(), base.c_str(),
              GripperContactDetector::mode_name(gripper_contact_config_.mode));
}

void OpenArmHW::update_gripper_contact(double motor_torque,
                                       double joint_position,
                                       double joint_velocity,
                                       double command_position, double dt) {
  if (!gripper_contact_ ||
      gripper_contact_config_.mode == GripperContactMode::kDisabled) {
    return;
  }
  const bool sensor_valid = std::isfinite(motor_torque);
  gripper_contact_->update(motor_torque, joint_position, joint_velocity,
                           command_position, dt, sensor_valid);
  log_gripper_close_summary(command_position, dt);
}

void OpenArmHW::publish_gripper_diagnostics() {
  if (!gripper_contact_ || contact_state_pub_ == nullptr ||
      motor_torque_pub_ == nullptr) {
    return;
  }
  std_msgs::msg::Int32 state_msg;
  state_msg.data = static_cast<int>(gripper_contact_->state());
  contact_state_pub_->publish(state_msg);

  std_msgs::msg::Float64 torque_msg;
  torque_msg.data = gripper_contact_->filtered_torque();
  motor_torque_pub_->publish(torque_msg);
}

void OpenArmHW::log_gripper_contact_throttled() {
  if (!gripper_contact_ ||
      gripper_contact_config_.mode == GripperContactMode::kDisabled) {
    return;
  }
  const double period = gripper_log_rate_hz_ > 0.0 ? 1.0 / gripper_log_rate_hz_
                                                   : 0.1;
  const auto now = rclcpp::Clock(RCL_ROS_TIME).now();
  if (last_gripper_log_time_.seconds() > 0.0 &&
      (now - last_gripper_log_time_).seconds() < period) {
    return;
  }
  last_gripper_log_time_ = now;
  RCLCPP_INFO(
      rclcpp::get_logger("OpenArmHW"),
      "[gripper-contact] state=%s result=%s raw_tau=%.3f filtered=%.3f "
      "bias=%.3f delta=%.3f peak=%.3f Nm hold_pos=%.4f counter=%d",
      GripperContactDetector::state_name(gripper_contact_->state()),
      GripperContactDetector::result_name(gripper_contact_->close_result()),
      gripper_contact_->raw_torque(), gripper_contact_->filtered_torque(),
      gripper_contact_->torque_bias(), gripper_contact_->contact_delta(),
      gripper_contact_->peak_torque(), gripper_contact_->hold_position(),
      gripper_contact_->contact_counter());
}

void OpenArmHW::log_gripper_close_summary(double target_position,
                                          double /*dt*/) {
  if (!gripper_contact_) {
    return;
  }
  const GripperCloseResult result = gripper_contact_->close_result();
  if (result == GripperCloseResult::kNone || result == last_reported_result_) {
    return;  // nothing new (or already reported)
  }
  last_reported_result_ = result;
  RCLCPP_WARN(
      rclcpp::get_logger("OpenArmHW"),
      "[gripper-contact] CLOSE SETTLED: result=%s state=%s peak_torque=%.3f "
      "Nm hold_pos=%.4f target=%.4f elapsed=%.2fs",
      GripperContactDetector::result_name(result),
      GripperContactDetector::state_name(gripper_contact_->state()),
      gripper_contact_->peak_torque(), gripper_contact_->hold_position(),
      target_position, gripper_contact_->close_elapsed_sec());
}

void OpenArmHW::return_to_zero() {
  RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Returning to zero position...");

  // Keep the gripper open while the arm returns home.  The old implementation
  // sent +0.044 directly as a motor angle.  That value is a parallel-gripper
  // joint displacement, is outside the v2 pinch limits [-0.7854, 0], and was
  // not converted to motor radians for the v1 parallel gripper either.
  const double gripper_home_joint =
      ee_type_ == "pinch_gripper" ? -0.35 : GRIPPER_JOINT_0_POSITION;
  const double gripper_home_motor =
      joint_to_motor_radians(gripper_home_joint);

  openarm_->refresh_all();
  // Return arm to zero with MIT control
  std::vector<openarm::damiao_motor::MITParam> arm_params;
  for (size_t i = 0; i < ARM_DOF; ++i) {
    arm_params.push_back({kp_[i], kd_[i], 0.0, 0.0, 0.0});
  }
  openarm_->get_arm().mit_control_all(arm_params);

  // Return gripper to zero if enabled
  if (hand_) {
    openarm_->get_gripper().mit_control_all(
        {{gripper_kp_, gripper_kd_, gripper_home_motor, 0.0, 0.0}});
  }
  std::this_thread::sleep_for(std::chrono::microseconds(1000));
  openarm_->recv_all();
  const auto& arm_motors = openarm_->get_arm().get_motors();

  std::vector<double> start_pos(ARM_DOF, 0.0);
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    start_pos[i] = arm_motors[i].get_position();
  }

  const int steps = 200;
  const int step_ms = 10;

  for (int step = 0; step <= steps; ++step) {
    double t = static_cast<double>(step) / steps;  // 0.0 → 1.0

    std::vector<openarm::damiao_motor::MITParam> arm_params;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      double target = start_pos[i] + t * (ZERO_POSITION[i] - start_pos[i]);
      arm_params.push_back({kp_[i], kd_[i], target, 0.0, 0.0});
    }
    openarm_->get_arm().mit_control_all(arm_params);

    if (hand_) {
      openarm_->get_gripper().mit_control_all(
          {{gripper_kp_, gripper_kd_, gripper_home_motor, 0.0, 0.0}});
    }

    openarm_->recv_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
  }

  RCLCPP_WARN(rclcpp::get_logger("OpenArmHW"),
              "Zero command sequence finished; actual arrival is NOT verified here. "
              "Require fresh motor feedback and pose acceptance before H2 or ACT.");
}

// void OpenArmHW::return_to_zero() {
//   RCLCPP_INFO(rclcpp::get_logger("OpenArmHW"), "Returning to zero
//   position...");

//   // Return arm to zero with MIT control
//   std::vector<openarm::damiao_motor::MITParam> arm_params;
//   for (size_t i = 0; i < ARM_DOF; ++i) {
//     arm_params.push_back({kp_[i], kd_[i], 0.0, 0.0, 0.0});
//   }
//   openarm_->get_arm().mit_control_all(arm_params);

//   // Return gripper to zero if enabled
//   if (hand_) {
//     openarm_->get_gripper().mit_control_all(
//         {{GRIPPER_KP, GRIPPER_KD, GRIPPER_JOINT_0_POSITION, 0.0, 0.0}});
//   }
//   std::this_thread::sleep_for(std::chrono::microseconds(1000));
//   openarm_->recv_all();
// }

double OpenArmHW::joint_to_motor_radians(double joint_value) {
  if (ee_type_ == "pinch_gripper") {
    // revolute: joint 0-1.5708 rad -> motor 0-1.5708
    return joint_value;
  } else {
    // parallel_link (prismatic): 0-0.044m -> 0 to -1.0472 rad
    return (joint_value / GRIPPER_JOINT_0_POSITION) * GRIPPER_MOTOR_1_RADIANS;
  }
}

double OpenArmHW::motor_radians_to_joint(double motor_radians) {
  if (ee_type_ == "pinch_gripper") {
    // revolute:
    return motor_radians;
  } else {
    // parallel_link (prismatic)
    return GRIPPER_JOINT_0_POSITION * (motor_radians / GRIPPER_MOTOR_1_RADIANS);
  }
}

// // Gripper mapping helper functions
// double OpenArmHW::joint_to_motor_radians(double joint_value) {
//   // Joint 0=closed -> motor 0 rad, Joint 0.044=open -> motor -1.0472 rad
//   return (joint_value / GRIPPER_JOINT_0_POSITION) *
//          GRIPPER_MOTOR_1_RADIANS;  // Scale from 0-0.044 to 0 to -1.0472
// }

// double OpenArmHW::motor_radians_to_joint(double motor_radians) {
//   // Motor 0 rad=closed -> joint 0, Motor -1.0472 rad=open -> joint 0.044
//   return GRIPPER_JOINT_0_POSITION *
//          (motor_radians /
//           GRIPPER_MOTOR_1_RADIANS);  // Scale from 0 to -1.0472 to 0-0.044
// }

}  // namespace openarm_hardware

// Software-only launch admission can inspect this exported symbol with nm
// without loading a hardware instance. Legacy installed binaries lack it.
extern "C" __attribute__((visibility("default"))) int
openarm_can_ownership_protocol_version() { return 1; }

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArmHW,
                       hardware_interface::SystemInterface)
