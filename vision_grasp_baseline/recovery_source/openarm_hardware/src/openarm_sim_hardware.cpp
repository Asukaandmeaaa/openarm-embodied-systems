// Copyright 2026 Enactic, Inc.
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

#include "openarm_hardware/openarm_sim_hardware.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace openarm_hardware {

double OpenArmSimHW::closing_direction() const {
  // v1.x parallel_link prismatic: 0.044 open -> 0 closed -> -1.
  if (ee_type_ == "parallel_link") {
    return -1.0;
  }
  // v2.0 revolute finger: 0 = closed. Right hand opens toward negative
  // (closing = joint increasing -> +1); the left hand is the mirror image and
  // opens toward positive (closing = joint decreasing -> -1).
  return (arm_prefix_ == "left_") ? -1.0 : 1.0;
}

bool OpenArmSimHW::parse_config(
    const hardware_interface::HardwareInfo& info) {
  auto it = info.hardware_parameters.find("arm_prefix");
  arm_prefix_ = (it != info.hardware_parameters.end()) ? it->second : "";
  it = info.hardware_parameters.find("ee_type");
  ee_type_ = (it != info.hardware_parameters.end()) ? it->second : "pinch_gripper";

  it = info.hardware_parameters.find("hand");
  if (it == info.hardware_parameters.end()) {
    hand_ = true;
  } else {
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    hand_ = (value == "true");
  }

  auto dbl = [&info, this](const std::string& key, double* out) {
    auto i = info.hardware_parameters.find(key);
    if (i != info.hardware_parameters.end()) {
      *out = std::stod(i->second);
    }
  };
  dbl("sim_arm_follow_tau", &arm_follow_tau_);
  dbl("sim_gripper_follow_tau", &gripper_follow_tau_);
  dbl("sim_gripper_kp", &gripper_kp_);
  dbl("sim_gripper_kd", &gripper_kd_);
  dbl("sim_gripper_block_position", &sim_block_position_);
  // sim_gripper_block_position is expressed in the right-hand joint frame
  // (negative = between open -0.35 and closed 0). The left hand is the mirror
  // image, so the same physical contact point is the negated coordinate.
  if (arm_prefix_ == "left_" && ee_type_ != "parallel_link") {
    sim_block_position_ = -sim_block_position_;
  }
  dbl("gripper_torque_threshold", &gripper_contact_config_.torque_threshold);
  dbl("gripper_torque_release_threshold",
      &gripper_contact_config_.torque_release_threshold);
  dbl("gripper_hard_torque_limit", &gripper_contact_config_.hard_torque_limit);
  dbl("gripper_torque_filter_alpha", &gripper_contact_config_.filter_alpha);
  dbl("gripper_hold_kp_scale", &gripper_contact_config_.hold_kp_scale);
  dbl("gripper_hold_kd_scale", &gripper_contact_config_.hold_kd_scale);
  dbl("gripper_hold_position_offset",
      &gripper_contact_config_.hold_position_offset);
  dbl("gripper_close_timeout", &gripper_contact_config_.max_close_duration_sec);
  dbl("gripper_position_reached_tolerance",
      &gripper_contact_config_.position_reached_tolerance);
  dbl("gripper_contact_log_rate", &gripper_log_rate_hz_);

  it = info.hardware_parameters.find("gripper_contact_confirm_cycles");
  if (it != info.hardware_parameters.end()) {
    gripper_contact_config_.confirm_cycles = std::stoi(it->second);
  }
  it = info.hardware_parameters.find("gripper_contact_mode");
  if (it != info.hardware_parameters.end()) {
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "enforce") {
      gripper_contact_config_.mode = GripperContactMode::kEnforce;
    } else if (value == "monitor") {
      gripper_contact_config_.mode = GripperContactMode::kMonitor;
    } else {
      gripper_contact_config_.mode = GripperContactMode::kDisabled;
    }
  }
  it = info.hardware_parameters.find("sim_gripper_block");
  if (it != info.hardware_parameters.end()) {
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    sim_block_ = (value == "true");
  }

  gripper_contact_config_.closing_direction = closing_direction();

  RCLCPP_INFO(rclcpp::get_logger("OpenArmSimHW"),
              "Sim config: prefix=%s ee=%s block=%s block_pos=%.4f "
              "gripper_kp=%.3f contact_mode=%s threshold=%.3f Nm",
              arm_prefix_.c_str(), ee_type_.c_str(),
              sim_block_ ? "enabled" : "disabled", sim_block_position_,
              gripper_kp_,
              GripperContactDetector::mode_name(gripper_contact_config_.mode),
              gripper_contact_config_.torque_threshold);
  return true;
}

void OpenArmSimHW::generate_joint_names() {
  joint_names_.clear();
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    joint_names_.push_back("openarm_" + arm_prefix_ + "joint" +
                           std::to_string(i));
  }
  if (hand_) {
    joint_names_.push_back("openarm_" + arm_prefix_ + "finger_joint1");
  }
}

hardware_interface::CallbackReturn OpenArmSimHW::on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) {
  if (hardware_interface::SystemInterface::on_init(params) !=
      CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  if (!parse_config(params.hardware_info)) {
    return CallbackReturn::ERROR;
  }
  generate_joint_names();

  const size_t total_joints = joint_names_.size();
  pos_commands_.assign(total_joints, 0.0);
  vel_commands_.assign(total_joints, 0.0);
  tau_commands_.assign(total_joints, 0.0);
  pos_states_.assign(total_joints, 0.0);
  vel_states_.assign(total_joints, 0.0);
  tau_states_.assign(total_joints, 0.0);

  // Arm joints start at 0. The gripper is set to OPEN in on_configure() (the
  // URDF initial_value would otherwise leave it at 0 = closed).

  gripper_contact_ = std::make_unique<GripperContactDetector>();
  gripper_contact_->configure(gripper_contact_config_);
  setup_gripper_diagnostics();

  RCLCPP_INFO(rclcpp::get_logger("OpenArmSimHW"),
              "OpenArmSimHW initialized with %zu joints", total_joints);
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArmSimHW::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Start the gripper OPEN so a close trajectory is observable (a real robot
  // powers up with the gripper open too). pinch_gripper (v2.0 revolute):
  // right hand opens negative, left hand opens positive; parallel_link (v1.x
  // prismatic) opens positive.
  if (hand_ && joint_names_.size() > ARM_DOF) {
    double open_pos = -0.35;
    if (ee_type_ == "parallel_link") {
      open_pos = 0.044;
    } else if (arm_prefix_ == "left_") {
      open_pos = 0.35;
    }
    pos_states_[ARM_DOF] = open_pos;
    vel_states_[ARM_DOF] = 0.0;
    tau_states_[ARM_DOF] = 0.0;
    // Command interfaces start at 0.0 (which means "closed") until a
    // controller is loaded; seed the gripper command with the open position so
    // the hardware does not slam shut (or trip contact) during bring-up.
    pos_commands_[ARM_DOF] = open_pos;
  }
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
OpenArmSimHW::export_state_interfaces() {
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
OpenArmSimHW::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
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

hardware_interface::CallbackReturn OpenArmSimHW::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArmSimHW"), "Activating...");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArmSimHW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArmSimHW"), "Deactivating...");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type OpenArmSimHW::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // State is updated in write(); nothing to do on the read side.
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OpenArmSimHW::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  const double dt = std::max(period.seconds(), 1e-6);
  const double dir = closing_direction();

  // Arm joints: first-order lag toward the commanded position.
  const double arm_k = 1.0 - std::exp(-dt / std::max(arm_follow_tau_, 1e-3));
  for (size_t i = 0; i < ARM_DOF; ++i) {
    const double old = pos_states_[i];
    pos_states_[i] += arm_k * (pos_commands_[i] - old);
    vel_states_[i] = (pos_states_[i] - old) / dt;
    tau_states_[i] = 5.0 * (pos_commands_[i] - pos_states_[i]);
  }

  if (hand_ && joint_names_.size() > ARM_DOF) {
    // Effective command already accounts for a latched contact hold.
    double command = pos_commands_[ARM_DOF];
    double gkp = gripper_kp_;
    double gkd = gripper_kd_;
    if (gripper_contact_) {
      command = gripper_contact_->effective_command(command);
      gkp *= gripper_contact_->kp_scale();
      gkd *= gripper_contact_->kd_scale();
    }

    // Spring-damper around the (possibly held) target.
    const double error = (command - pos_states_[ARM_DOF]) * dir;
    double torque = gkp * error - gkd * vel_states_[ARM_DOF];

    // Contact/block model: if the object blocks before the target and the
    // command still demands closure, freeze the position at the block point.
    const double block_close = sim_block_position_ * dir;
    const double pos_close = pos_states_[ARM_DOF] * dir;
    if (sim_block_ && pos_close >= block_close &&
        (command - pos_states_[ARM_DOF]) * dir > 1e-6) {
      pos_states_[ARM_DOF] = sim_block_position_;
      vel_states_[ARM_DOF] = 0.0;
      torque = gkp * (command - pos_states_[ARM_DOF]) * dir;
    } else {
      const double old = pos_states_[ARM_DOF];
      const double k = 1.0 - std::exp(-dt / std::max(gripper_follow_tau_, 1e-3));
      pos_states_[ARM_DOF] += k * (command - old);
      vel_states_[ARM_DOF] = (pos_states_[ARM_DOF] - old) / dt;
    }
    tau_states_[ARM_DOF] = torque;

    // Feed the simulated motor torque into the contact detector.
    if (gripper_contact_) {
      gripper_contact_->update(tau_states_[ARM_DOF], pos_states_[ARM_DOF],
                               vel_states_[ARM_DOF], pos_commands_[ARM_DOF], dt,
                               std::isfinite(tau_states_[ARM_DOF]));
    }
    publish_gripper_diagnostics();
    log_gripper_contact_throttled();
    log_gripper_close_summary();
  }

  return hardware_interface::return_type::OK;
}

void OpenArmSimHW::setup_gripper_diagnostics() {
  if (gripper_contact_config_.mode == GripperContactMode::kDisabled) {
    return;
  }
  diag_node_ =
      std::make_shared<rclcpp::Node>("openarm_sim_hw_diagnostics_" + arm_prefix_);
  const std::string base = arm_prefix_ + "gripper";
  contact_state_pub_ = diag_node_->create_publisher<std_msgs::msg::Int32>(
      base + "/contact_state", rclcpp::SensorDataQoS());
  motor_torque_pub_ = diag_node_->create_publisher<std_msgs::msg::Float64>(
      base + "/motor_torque", rclcpp::SensorDataQoS());
}

void OpenArmSimHW::publish_gripper_diagnostics() {
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

void OpenArmSimHW::log_gripper_contact_throttled() {
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
      rclcpp::get_logger("OpenArmSimHW"),
      "[gripper-contact] state=%s result=%s raw_tau=%.3f filtered=%.3f "
      "bias=%.3f delta=%.3f peak=%.3f Nm hold_pos=%.4f counter=%d",
      GripperContactDetector::state_name(gripper_contact_->state()),
      GripperContactDetector::result_name(gripper_contact_->close_result()),
      gripper_contact_->raw_torque(), gripper_contact_->filtered_torque(),
      gripper_contact_->torque_bias(), gripper_contact_->contact_delta(),
      gripper_contact_->peak_torque(), gripper_contact_->hold_position(),
      gripper_contact_->contact_counter());
}

void OpenArmSimHW::log_gripper_close_summary() {
  if (!gripper_contact_) {
    return;
  }
  const GripperCloseResult result = gripper_contact_->close_result();
  if (result == GripperCloseResult::kNone || result == last_reported_result_) {
    return;
  }
  last_reported_result_ = result;
  RCLCPP_WARN(
      rclcpp::get_logger("OpenArmSimHW"),
      "[gripper-contact] CLOSE SETTLED: result=%s state=%s peak_torque=%.3f "
      "Nm hold_pos=%.4f elapsed=%.2fs",
      GripperContactDetector::result_name(result),
      GripperContactDetector::state_name(gripper_contact_->state()),
      gripper_contact_->peak_torque(), gripper_contact_->hold_position(),
      gripper_contact_->close_elapsed_sec());
}

}  // namespace openarm_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArmSimHW,
                       hardware_interface::SystemInterface)
