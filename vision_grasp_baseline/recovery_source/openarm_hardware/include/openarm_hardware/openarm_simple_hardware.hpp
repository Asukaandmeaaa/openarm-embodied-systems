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

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "openarm_hardware/dynamics.hpp"
#include "openarm_hardware/can_ownership.hpp"
#include "openarm_hardware/gripper_contact_detector.hpp"
#include "openarm_hardware/visibility_control.h"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace openarm_hardware {

/**
 * @brief Simplified OpenArm V10 Hardware Interface
 *
 * This is a simplified version that uses the OpenArm CAN API directly,
 * following the pattern from full_arm.cpp example. Much simpler than
 * the original implementation.
 */
class OpenArmHW : public hardware_interface::SystemInterface {
 public:
  OpenArmHW();
  ~OpenArmHW() override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_init(
      const hardware_interface::HardwareComponentInterfaceParams& params)
      override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces()
      override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces()
      override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::return_type read(const rclcpp::Time& time,
                                       const rclcpp::Duration& period) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::return_type write(
      const rclcpp::Time& time, const rclcpp::Duration& period) override;

 private:
  // V10 default configuration
  static constexpr size_t ARM_DOF = 7;
  static constexpr bool ENABLE_GRIPPER = true;

  // Default motor configuration for V10
  const std::vector<openarm::damiao_motor::MotorType> DEFAULT_MOTOR_TYPES = {
      openarm::damiao_motor::MotorType::DM8009,  // Joint 1
      openarm::damiao_motor::MotorType::DM8009,  // Joint 2
      openarm::damiao_motor::MotorType::DM4340,  // Joint 3
      openarm::damiao_motor::MotorType::DM4340,  // Joint 4
      openarm::damiao_motor::MotorType::DM4310,  // Joint 5
      openarm::damiao_motor::MotorType::DM4310,  // Joint 6
      openarm::damiao_motor::MotorType::DM4310   // Joint 7
  };

  const std::vector<uint32_t> DEFAULT_SEND_CAN_IDS = {0x01, 0x02, 0x03, 0x04,
                                                      0x05, 0x06, 0x07};
  const std::vector<uint32_t> DEFAULT_RECV_CAN_IDS = {0x11, 0x12, 0x13, 0x14,
                                                      0x15, 0x16, 0x17};

  const openarm::damiao_motor::MotorType DEFAULT_GRIPPER_MOTOR_TYPE =
      openarm::damiao_motor::MotorType::DM4310;
  const uint32_t DEFAULT_GRIPPER_SEND_CAN_ID = 0x08;
  const uint32_t DEFAULT_GRIPPER_RECV_CAN_ID = 0x18;

  // Gains
  std::vector<double> kp_ = {70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0};
  std::vector<double> kd_ = {2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5};

  const double GRIPPER_JOINT_0_POSITION = 0.044;
  const double GRIPPER_JOINT_1_POSITION = 0.0;
  const double GRIPPER_MOTOR_0_RADIANS = 0.0;
  const double GRIPPER_MOTOR_1_RADIANS = -1.0472;
  const double GRIPPER_KP = 5.0;
  const double GRIPPER_KD = 0.1;

  double gripper_kp_ = GRIPPER_KP;
  double gripper_kd_ = GRIPPER_KD;

  // Configuration
  std::string can_interface_;
  std::string arm_prefix_;
  std::string ee_type_;
  bool hand_;
  bool can_fd_;
  bool startup_hold_current_{false};
  hardware_interface::CallbackReturn activate_hold_current();

  // Gravity compensation (KDL feedforward, default off)
  bool gravity_compensation_{false};
  double gravity_scale_{1.0};
  std::string gravity_urdf_path_;
  std::string gravity_base_link_;
  std::string gravity_tip_link_;
  std::unique_ptr<Dynamics> dynamics_;
  std::vector<double> gravity_ff_;

  // Declared before OpenArm so its lock outlives socket destruction. Keep this
  // for the entire hardware-instance lifetime, including deactivate/cleanup.
  std::unique_ptr<CanOwnership> can_ownership_;
  // OpenArm instance
  std::unique_ptr<openarm::can::socket::OpenArm> openarm_;

  // Generated joint names for this arm instance
  std::vector<std::string> joint_names_;

  // ROS2 control state and command vectors
  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> tau_commands_;
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;

  // Hardware-level leader mode.  In passive mode J1-J7 MIT position gain is
  // zero while velocity damping and gravity feedforward remain active.  J8
  // remains electrically controlled.  The heartbeat makes loss of the
  // userspace teleop node fail back to rigid mode.
  std::atomic<bool> passive_requested_{false};
  std::atomic<bool> gravity_ready_{false};
  std::atomic<int64_t> passive_heartbeat_ns_{0};
  bool passive_applied_{false};
  double passive_heartbeat_timeout_{0.5};
  rclcpp::Node::SharedPtr control_node_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr passive_service_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr passive_heartbeat_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr passive_state_pub_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> control_executor_;
  std::thread control_thread_;

  static constexpr std::array<double, ARM_DOF> ZERO_POSITION = {
      0.0,  // joint1
      0.0,  // joint2
      0.0,  // joint3
      0.0,  // joint4
      0.0,  // joint5
      0.0,  // joint6
      0.0,  // joint7
  };

  // Helper methods
  void return_to_zero();
  bool parse_config(const hardware_interface::HardwareInfo& info);
  void generate_joint_names();
  void setup_passive_control();
  void shutdown_passive_control();
  void set_passive_requested(bool requested);
  void publish_passive_state(bool enabled);

  // Gripper mapping functions
  double joint_to_motor_radians(double joint_value);
  double motor_radians_to_joint(double motor_radians);

  // Gripper contact detection (shared with the simulation plugin)
  GripperContactConfig gripper_contact_config_;
  std::unique_ptr<GripperContactDetector> gripper_contact_;
  rclcpp::Node::SharedPtr diag_node_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr contact_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr motor_torque_pub_;
  rclcpp::Time last_gripper_log_time_{0, 0, RCL_ROS_TIME};
  double gripper_log_rate_hz_{10.0};
  GripperCloseResult last_reported_result_{GripperCloseResult::kNone};

  void setup_gripper_diagnostics();
  void update_gripper_contact(double motor_torque, double joint_position,
                              double joint_velocity, double command_position,
                              double dt);
  void publish_gripper_diagnostics();
  void log_gripper_contact_throttled();
  void log_gripper_close_summary(double target_position, double dt);
};

}  // namespace openarm_hardware
