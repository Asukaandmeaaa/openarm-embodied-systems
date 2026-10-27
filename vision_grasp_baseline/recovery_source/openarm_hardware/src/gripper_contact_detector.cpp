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

#include "openarm_hardware/gripper_contact_detector.hpp"

#include <algorithm>
#include <cmath>

namespace openarm_hardware {

const char* GripperContactDetector::mode_name(GripperContactMode mode) {
  switch (mode) {
    case GripperContactMode::kDisabled:
      return "disabled";
    case GripperContactMode::kMonitor:
      return "monitor";
    case GripperContactMode::kEnforce:
      return "enforce";
  }
  return "unknown";
}

const char* GripperContactDetector::state_name(GripperContactState state) {
  switch (state) {
    case GripperContactState::kDisabled:
      return "DISABLED";
    case GripperContactState::kIdle:
      return "IDLE";
    case GripperContactState::kClosing:
      return "CLOSING";
    case GripperContactState::kContactCandidate:
      return "CONTACT_CANDIDATE";
    case GripperContactState::kContactLatched:
      return "CONTACT_LATCHED";
    case GripperContactState::kHardLimit:
      return "HARD_LIMIT";
    case GripperContactState::kTimeout:
      return "TIMEOUT";
  }
  return "UNKNOWN";
}

const char* GripperContactDetector::result_name(GripperCloseResult result) {
  switch (result) {
    case GripperCloseResult::kNone:
      return "NONE";
    case GripperCloseResult::kPositionReached:
      return "POSITION_REACHED";
    case GripperCloseResult::kContactDetected:
      return "CONTACT_DETECTED";
    case GripperCloseResult::kTimeout:
      return "TIMEOUT";
    case GripperCloseResult::kHardTorqueLimit:
      return "HARD_TORQUE_LIMIT";
    case GripperCloseResult::kSensorInvalid:
      return "SENSOR_INVALID";
  }
  return "UNKNOWN";
}

void GripperContactDetector::configure(const GripperContactConfig& config) {
  config_ = config;
  reset();
  if (config_.mode == GripperContactMode::kDisabled) {
    state_ = GripperContactState::kDisabled;
  }
}

void GripperContactDetector::reset() {
  state_ = GripperContactState::kIdle;
  if (config_.mode == GripperContactMode::kDisabled) {
    state_ = GripperContactState::kDisabled;
  }
  close_result_ = GripperCloseResult::kNone;
  raw_torque_ = 0.0;
  filtered_torque_ = 0.0;
  torque_bias_ = 0.0;
  contact_delta_ = 0.0;
  peak_torque_ = 0.0;
  hold_position_ = 0.0;
  contact_counter_ = 0;
  bias_samples_ = 0;
  bias_accum_ = 0.0;
  close_elapsed_sec_ = 0.0;
  close_start_position_ = 0.0;
  last_command_position_ = 0.0;
  filter_initialized_ = false;
}

bool GripperContactDetector::closing_condition(
    double command_position, double joint_position) const {
  // Closing means the command demands more closure than the current position.
  const double gap =
      (command_position - joint_position) * config_.closing_direction;
  return gap > config_.closing_command_gap;
}

void GripperContactDetector::set_result(GripperCloseResult result) {
  if (close_result_ != GripperCloseResult::kNone) {
    return;  // first settled result wins
  }
  close_result_ = result;
}

void GripperContactDetector::begin_closing(double joint_position) {
  state_ = GripperContactState::kClosing;
  contact_counter_ = 0;
  close_elapsed_sec_ = 0.0;
  bias_samples_ = 0;
  bias_accum_ = 0.0;
  torque_bias_ = filtered_torque_;  // fallback baseline
  close_start_position_ = joint_position;
}

void GripperContactDetector::update_bias(double filtered) {
  // First N cycles of closing capture the no-contact baseline. This only runs
  // before any contact is seen; after latch the bias is frozen.
  if (state_ != GripperContactState::kClosing &&
      state_ != GripperContactState::kContactCandidate) {
    return;
  }
  if (bias_samples_ < config_.bias_sample_cycles) {
    bias_accum_ += filtered;
    ++bias_samples_;
    if (bias_samples_ >= config_.bias_sample_cycles) {
      torque_bias_ = bias_accum_ / static_cast<double>(config_.bias_sample_cycles);
    }
  }
}

void GripperContactDetector::latch_contact(double joint_position) {
  state_ = GripperContactState::kContactLatched;
  contact_counter_ = config_.confirm_cycles;
  hold_position_ = joint_position;
  set_result(GripperCloseResult::kContactDetected);
}

void GripperContactDetector::trip_hard_limit(double joint_position) {
  state_ = GripperContactState::kHardLimit;
  hold_position_ = joint_position;
  set_result(GripperCloseResult::kHardTorqueLimit);
}

void GripperContactDetector::update(double raw_torque, double joint_position,
                                    double joint_velocity,
                                    double command_position, double dt,
                                    bool sensor_valid) {
  raw_torque_ = raw_torque;
  peak_torque_ = std::max(peak_torque_, std::fabs(raw_torque));
  (void)joint_velocity;  // reserved for future hold-time damping
  const double prev_command = last_command_position_;
  last_command_position_ = command_position;

  if (config_.mode == GripperContactMode::kDisabled) {
    state_ = GripperContactState::kDisabled;
    return;
  }

  // EMA filter (alpha=1 => pass-through)
  const double alpha = std::clamp(config_.filter_alpha, 0.0, 1.0);
  if (!filter_initialized_) {
    filtered_torque_ = raw_torque;
    filter_initialized_ = true;
  } else {
    filtered_torque_ =
        alpha * raw_torque + (1.0 - alpha) * filtered_torque_;
  }

  const bool is_closing = closing_condition(command_position, joint_position);

  // Reopen: command demands opening beyond the current position -> reset the
  // whole detector so the next close starts clean.
  const double reopen_gap =
      (command_position - joint_position) * config_.closing_direction;
  if (reopen_gap < -config_.reopen_margin) {
    reset();
    return;
  }

  // Sensor validity is a safety gate while closing.
  if (!sensor_valid && state_ != GripperContactState::kIdle &&
      state_ != GripperContactState::kDisabled) {
    set_result(GripperCloseResult::kSensorInvalid);
    state_ = GripperContactState::kIdle;
    return;
  }

  // Terminal states keep holding until reopen.
  if (state_ == GripperContactState::kContactLatched ||
      state_ == GripperContactState::kHardLimit ||
      state_ == GripperContactState::kTimeout) {
    return;
  }

  // Transition into closing.
  if (state_ == GripperContactState::kIdle) {
    if (is_closing) {
      begin_closing(joint_position);
    } else {
      return;
    }
  }

  // Active closing states: check the outcome signals.
  if (state_ == GripperContactState::kClosing ||
      state_ == GripperContactState::kContactCandidate) {
    // Position reached (no contact): the commanded close position was actually
    // achieved, e.g. empty close or a soft block that still yields. This only
    // settles once the gripper really closed by a minimum amount AND the
    // command has stopped moving; otherwise the transient "command == position"
    // at trajectory start (or a mid-trajectory tracking pass) would settle a
    // bogus result before a real contact can be seen.
    const double closed_motion =
        (joint_position - close_start_position_) * config_.closing_direction;
    if (std::fabs(command_position - joint_position) <=
            config_.position_reached_tolerance &&
        closed_motion >= kMinCloseMotion &&
        std::fabs(command_position - prev_command) <= kCommandStaticEps) {
      state_ = GripperContactState::kIdle;
      set_result(GripperCloseResult::kPositionReached);
      return;
    }

    // Stopped closing before any result: return to idle and drop the attempt.
    if (!is_closing) {
      state_ = GripperContactState::kIdle;
      return;
    }

    // Track close duration for the timeout.
    close_elapsed_sec_ += std::max(0.0, dt);
    if (config_.max_close_duration_sec > 0.0 &&
        close_elapsed_sec_ >= config_.max_close_duration_sec) {
      state_ = GripperContactState::kTimeout;
      set_result(GripperCloseResult::kTimeout);
      return;
    }

    // Capture the online baseline during the first cycles of closing.
    if (config_.use_online_bias) {
      update_bias(filtered_torque_);
    }

    // Contact signal with hysteresis.
    const double delta = filtered_torque_ - torque_bias_;
    contact_delta_ = config_.use_absolute_contact_delta
                         ? std::fabs(delta)
                         : config_.closing_direction * delta;

    const bool over_trigger =
        config_.torque_threshold > 0.0 &&
        contact_delta_ >= config_.torque_threshold;
    const bool below_release =
        config_.torque_release_threshold > 0.0 &&
        contact_delta_ <= config_.torque_release_threshold;

    // Absolute hard safety limit trips immediately, no confirmation. It is
    // checked against RAW torque (not the EMA-filtered value) so a genuine
    // stall trips on the very first cycle; the limit itself is set well above
    // normal noise / friction so single-cycle noise can never trip it.
    if (config_.hard_torque_limit > 0.0 &&
        std::fabs(raw_torque_ - torque_bias_) >= config_.hard_torque_limit) {
      trip_hard_limit(joint_position);
      return;
    }

    if (over_trigger) {
      ++contact_counter_;
      state_ = GripperContactState::kContactCandidate;
      if (contact_counter_ >= config_.confirm_cycles) {
        latch_contact(joint_position);
      }
    } else if (below_release || contact_counter_ > 0) {
      contact_counter_ = 0;
      state_ = GripperContactState::kClosing;
    }
  }
}

double GripperContactDetector::effective_command(double command_position) const {
  if (config_.mode == GripperContactMode::kEnforce) {
    if (state_ == GripperContactState::kContactLatched ||
        state_ == GripperContactState::kHardLimit) {
      // The offset is a positive squeeze magnitude. Apply it in the configured
      // closing direction so mirrored left/right pinch grippers and the
      // parallel gripper all squeeze rather than accidentally reopen.
      return hold_position_ +
             config_.closing_direction * config_.hold_position_offset;
    }
  }
  return command_position;
}

double GripperContactDetector::kp_scale() const {
  if (config_.mode == GripperContactMode::kEnforce &&
      (state_ == GripperContactState::kContactLatched ||
       state_ == GripperContactState::kHardLimit)) {
    return config_.hold_kp_scale;
  }
  return 1.0;
}

double GripperContactDetector::kd_scale() const {
  if (config_.mode == GripperContactMode::kEnforce &&
      (state_ == GripperContactState::kContactLatched ||
       state_ == GripperContactState::kHardLimit)) {
    return config_.hold_kd_scale;
  }
  return 1.0;
}

}  // namespace openarm_hardware
