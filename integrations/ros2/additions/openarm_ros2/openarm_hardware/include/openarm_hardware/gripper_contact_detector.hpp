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

// GripperContactDetector: pure contact-detection state machine for the OpenArm
// single-DOF gripper. It consumes raw motor-side torque, filters it, compares
// it against a contact threshold (with hysteresis and confirmation cycles) and
// latches a contact hold target. All logic is independent of CAN / ros2_control
// so it can be unit tested without hardware and shared by the real hardware
// plugin (OpenArmHW) and the simulation plugin (OpenArmSimHW).
//
// The torque value is MOTOR-SIDE torque in Nm (Damiao MIT state tau), NOT a
// fingertip force in Newtons. The contact threshold must be calibrated on the
// real robot; until then the detector is meant to run in monitor mode.

#pragma once

#include <cstdint>
#include <string>

namespace openarm_hardware {

enum class GripperContactMode : int {
  kDisabled = 0,  // fully legacy behavior, no detection
  kMonitor = 1,   // detect + log + publish, never override command
  kEnforce = 2,   // detect + latch + hold position (torque-stop)
};

enum class GripperContactState : int {
  kDisabled = 0,
  kIdle = 1,             // not currently closing
  kClosing = 2,          // closing in progress, no contact yet
  kContactCandidate = 3, // over threshold, confirming
  kContactLatched = 4,   // contact confirmed, holding
  kHardLimit = 5,        // absolute hard torque limit reached
  kTimeout = 6,          // close timed out without contact/goal
};

enum class GripperCloseResult : int {
  kNone = 0,
  kPositionReached = 1,  // commanded close position actually reached
  kContactDetected = 2,  // torque contact confirmed
  kTimeout = 3,          // max close duration exceeded
  kHardTorqueLimit = 4,  // hard torque limit tripped
  kSensorInvalid = 5,    // torque feedback invalid while closing
};

struct GripperContactConfig {
  GripperContactMode mode{GripperContactMode::kDisabled};

  // Motor-side torque thresholds (Nm). TBD values must be calibrated on the
  // real robot; zero disables the corresponding check.
  double torque_threshold{0.0};          // contact trigger
  double torque_release_threshold{0.0};  // hysteresis release (<= trigger)
  double hard_torque_limit{0.0};         // absolute safety limit, 0 = off

  // EMA filter coefficient (0, 1]
  double filter_alpha{0.2};
  // Consecutive over-threshold cycles required before latching contact
  int confirm_cycles{5};

  // Sign of "closing" motion in joint units. OpenArm v2.0 revolute finger:
  // joint increases toward 0 when closing -> +1. Parallel-link prismatic
  // (0.044 open -> 0 closed) -> -1.
  double closing_direction{1.0};
  // Minimum |command - position| * direction to consider the gripper "closing".
  double closing_command_gap{0.0};
  // Reached the commanded close position within this tolerance (joint units).
  double position_reached_tolerance{0.005};
  // Reopen detection margin (joint units); command open beyond this resets.
  double reopen_margin{0.005};

  // Hold behavior after contact latch (enforce mode)
  double hold_kp_scale{0.5};
  double hold_kd_scale{1.0};
  // Positive extra squeeze magnitude beyond contact, applied along
  // closing_direction.
  double hold_position_offset{0.0};

  // Close timeout (s). Only counts while actively closing.
  double max_close_duration_sec{3.0};

  // Baseline / bias handling
  bool use_online_bias{true};  // sample baseline at close start
  int bias_sample_cycles{10};

  // Use |filtered - bias| as the contact signal (safe before sign is known).
  bool use_absolute_contact_delta{true};
};

class GripperContactDetector {
 public:
  GripperContactDetector() = default;

  void configure(const GripperContactConfig& config);
  const GripperContactConfig& config() const { return config_; }

  // Feed one control cycle. Call every write() while the gripper is active.
  //   raw_torque:      raw motor-side torque (Nm)
  //   joint_position:  current gripper joint position (joint units)
  //   joint_velocity:  current gripper joint velocity (joint units/s)
  //   command_position: commanded gripper joint position (joint units)
  //   dt:              control period (s)
  //   sensor_valid:    false if torque feedback is unavailable/NaN
  void update(double raw_torque, double joint_position, double joint_velocity,
              double command_position, double dt, bool sensor_valid = true);

  // Explicit reset (e.g. on hardware (re)configure).
  void reset();

  // --- command output (enforce mode) ---
  // Position command that should be sent to the motor this cycle.
  double effective_command(double command_position) const;
  // Gain scaling to apply once contact is latched (enforce mode).
  double kp_scale() const;
  double kd_scale() const;

  // --- status ---
  GripperContactState state() const { return state_; }
  GripperCloseResult close_result() const { return close_result_; }
  bool contact() const {
    return state_ == GripperContactState::kContactLatched;
  }
  bool hard_limit() const { return state_ == GripperContactState::kHardLimit; }
  bool timeout() const { return state_ == GripperContactState::kTimeout; }
  bool closing() const { return state_ == GripperContactState::kClosing ||
                                state_ == GripperContactState::kContactCandidate; }
  bool settled() const { return close_result_ != GripperCloseResult::kNone; }

  // --- diagnostics ---
  double raw_torque() const { return raw_torque_; }
  double filtered_torque() const { return filtered_torque_; }
  double torque_bias() const { return torque_bias_; }
  double contact_delta() const { return contact_delta_; }
  double peak_torque() const { return peak_torque_; }
  double hold_position() const { return hold_position_; }
  int contact_counter() const { return contact_counter_; }
  double close_elapsed_sec() const { return close_elapsed_sec_; }

  static const char* state_name(GripperContactState state);
  static const char* result_name(GripperCloseResult result);
  static const char* mode_name(GripperContactMode mode);

 private:
  // POSITION_REACHED is only meaningful once the gripper has actually closed
  // by at least this much from where closing began (joint units). This rejects
  // the "command == position at trajectory start" transient.
  static constexpr double kMinCloseMotion = 0.01;
  // POSITION_REACHED also requires the command to have stopped moving: while a
  // trajectory is interpolating, the command changes every cycle by far more
  // than this, so a mid-trajectory tracking pass cannot settle the result.
  static constexpr double kCommandStaticEps = 1e-4;

  bool closing_condition(double command_position, double joint_position) const;
  void begin_closing(double joint_position);
  void update_bias(double filtered);
  void latch_contact(double joint_position);
  void trip_hard_limit(double joint_position);
  void set_result(GripperCloseResult result);

  GripperContactConfig config_;
  GripperContactState state_{GripperContactState::kIdle};
  GripperCloseResult close_result_{GripperCloseResult::kNone};

  double raw_torque_{0.0};
  double filtered_torque_{0.0};
  double torque_bias_{0.0};
  double contact_delta_{0.0};
  double peak_torque_{0.0};
  double hold_position_{0.0};
  int contact_counter_{0};
  int bias_samples_{0};
  double bias_accum_{0.0};
  double close_elapsed_sec_{0.0};
  double close_start_position_{0.0};
  double last_command_position_{0.0};
  bool filter_initialized_{false};
};

}  // namespace openarm_hardware
