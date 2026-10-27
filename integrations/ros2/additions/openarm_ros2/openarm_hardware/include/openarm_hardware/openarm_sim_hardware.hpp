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

#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "openarm_hardware/gripper_contact_detector.hpp"
#include "openarm_hardware/visibility_control.h"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace openarm_hardware {

/**
 * @brief Simulated OpenArm hardware for validating the gripper contact
 * detection / torque-stop pipeline without CAN hardware.
 *
 * The 7 arm joints follow position commands with a small first-order lag (the
 * exact dynamics do not matter for contact validation). The single-DOF
 * gripper runs a tiny spring-damper model: while it moves freely it tracks the
 * position command; when `sim_gripper_block` is set it physically stops at
 * `sim_gripper_block_position` and the simulated motor torque grows with the
 * position error, which is exactly the signal GripperContactDetector consumes.
 *
 * The contact detector (same GripperContactDetector as OpenArmHW) runs in the
 * configured mode (disabled / monitor / enforce) and, in enforce mode,
 * overrides the effective gripper command to hold the latched contact
 * position, mirroring the real hardware behavior.
 */
class OpenArmSimHW : public hardware_interface::SystemInterface {
 public:
  OpenArmSimHW() = default;

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
  hardware_interface::return_type write(const rclcpp::Time& time,
                                        const rclcpp::Duration& period) override;

 private:
  static constexpr size_t ARM_DOF = 7;

  std::string arm_prefix_;
  std::string ee_type_;
  bool hand_{true};

  // Simulation parameters
  double arm_follow_tau_{0.02};     // first-order lag for arm joints (s)
  double gripper_follow_tau_{0.02}; // first-order lag for free gripper motion (s)
  double gripper_kp_{5.0};          // simulated motor stiffness (Nm/unit)
  double gripper_kd_{0.05};         // simulated motor damping
  bool sim_block_{false};           // enable the contact/block model
  double sim_block_position_{-0.20};  // joint position where the object blocks

  // Joint names, commands and states (same layout as OpenArmHW)
  std::vector<std::string> joint_names_;
  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> tau_commands_;
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;

  // Gripper contact detection (shared logic with the real hardware plugin)
  GripperContactConfig gripper_contact_config_;
  std::unique_ptr<GripperContactDetector> gripper_contact_;
  rclcpp::Node::SharedPtr diag_node_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr contact_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr motor_torque_pub_;
  rclcpp::Time last_gripper_log_time_{0, 0, RCL_ROS_TIME};
  double gripper_log_rate_hz_{10.0};
  GripperCloseResult last_reported_result_{GripperCloseResult::kNone};

  double closing_direction() const;
  void generate_joint_names();
  bool parse_config(const hardware_interface::HardwareInfo& info);
  void setup_gripper_diagnostics();
  void publish_gripper_diagnostics();
  void log_gripper_contact_throttled();
  void log_gripper_close_summary();
};

}  // namespace openarm_hardware
