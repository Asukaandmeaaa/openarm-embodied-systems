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

// Unit tests for GripperContactDetector (pure logic, no CAN / ros2_control).

#include <algorithm>

#include <gtest/gtest.h>

#include "openarm_hardware/gripper_contact_detector.hpp"

namespace openarm_hardware {

namespace {
// Shared fixture: OpenArm v2.0 revolute gripper. Joint convention: 0 = closed,
// negative = open, closing = joint increasing toward 0 (closing_direction +1).
constexpr double kDt = 0.01;  // 100 Hz control cycle
constexpr double kOpen = -0.35;
constexpr double kClose = -0.10;
constexpr double kThreshold = 0.3;        // contact trigger torque (Nm)
constexpr double kRelease = 0.15;         // hysteresis release
constexpr double kHardLimit = 2.0;        // absolute safety limit
constexpr double kContactTorque = 1.0;    // sustained contact torque
constexpr double kEmptyTorque = 0.05;     // no-load torque

GripperContactConfig enforce_config() {
  GripperContactConfig c;
  c.mode = GripperContactMode::kEnforce;
  c.torque_threshold = kThreshold;
  c.torque_release_threshold = kRelease;
  c.hard_torque_limit = kHardLimit;
  c.filter_alpha = 0.5;
  c.confirm_cycles = 5;
  c.closing_direction = 1.0;
  c.closing_command_gap = 0.001;
  c.position_reached_tolerance = 0.005;
  c.reopen_margin = 0.005;
  c.hold_kp_scale = 0.5;
  c.hold_kd_scale = 1.0;
  c.max_close_duration_sec = 3.0;
  c.bias_sample_cycles = 5;
  c.use_absolute_contact_delta = true;
  return c;
}
}  // namespace

// 1. disabled -> never triggers, command passes through
TEST(GripperContactDetector, DisabledNeverTriggers) {
  GripperContactConfig c = enforce_config();
  c.mode = GripperContactMode::kDisabled;
  GripperContactDetector d;
  d.configure(c);
  for (int i = 0; i < 100; ++i) {
    d.update(kContactTorque, -0.06, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.state(), GripperContactState::kDisabled);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kNone);
  EXPECT_DOUBLE_EQ(d.effective_command(kClose), kClose);
  EXPECT_DOUBLE_EQ(d.kp_scale(), 1.0);
}

// 2. single torque spike -> no trigger
TEST(GripperContactDetector, SingleSpikeDoesNotTrigger) {
  GripperContactDetector d;
  d.configure(enforce_config());
  // Bias phase (5 cycles at empty torque), then one single spike.
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  d.update(kContactTorque, kOpen, 0.0, kClose, kDt);
  for (int i = 0; i < 20; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  EXPECT_NE(d.state(), GripperContactState::kContactLatched);
  EXPECT_NE(d.close_result(), GripperCloseResult::kContactDetected);
}

// 3. sustained over-threshold -> CONTACT_DETECTED + hold position latched
TEST(GripperContactDetector, SustainedOverThresholdLatches) {
  GripperContactDetector d;
  d.configure(enforce_config());
  // Bias capture phase
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  // Sustained contact while position is blocked at -0.20 (between open and
  // close target): command keeps demanding kClose but position cannot follow.
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.state(), GripperContactState::kContactLatched);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kContactDetected);
  EXPECT_TRUE(d.contact());
  // Hold target is the contact position, command is overridden in enforce mode
  EXPECT_DOUBLE_EQ(d.hold_position(), -0.20);
  EXPECT_DOUBLE_EQ(d.effective_command(kClose), -0.20);
  EXPECT_DOUBLE_EQ(d.kp_scale(), 0.5);
}

TEST(GripperContactDetector, HoldOffsetFollowsPositiveClosingDirection) {
  GripperContactConfig c = enforce_config();
  c.hold_position_offset = 0.01;
  GripperContactDetector d;
  d.configure(c);
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  ASSERT_TRUE(d.contact());
  EXPECT_NEAR(d.effective_command(kClose), -0.19, 1e-12);
}

TEST(GripperContactDetector, HoldOffsetFollowsNegativeClosingDirection) {
  GripperContactConfig c = enforce_config();
  c.closing_direction = -1.0;
  c.hold_position_offset = 0.01;
  GripperContactDetector d;
  d.configure(c);
  constexpr double open = 0.35;
  constexpr double close = 0.10;
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, open, 0.0, close, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, 0.20, 0.0, close, kDt);
  }
  ASSERT_TRUE(d.contact());
  EXPECT_NEAR(d.effective_command(close), 0.19, 1e-12);
}

// 4. not closing (no command-position gap) -> never triggers
TEST(GripperContactDetector, NotClosingNeverTriggers) {
  GripperContactDetector d;
  d.configure(enforce_config());
  // Command equals position -> not closing even under high torque.
  for (int i = 0; i < 50; ++i) {
    d.update(kContactTorque, -0.20, 0.0, -0.20, kDt);
  }
  EXPECT_NE(d.state(), GripperContactState::kContactLatched);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kNone);
}

// 5. no-load torque below threshold -> no trigger
TEST(GripperContactDetector, EmptyTorqueNoTrigger) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 100; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  EXPECT_NE(d.close_result(), GripperCloseResult::kContactDetected);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kNone);
}

// 6. non-zero bias is subtracted before thresholding
TEST(GripperContactDetector, BiasIsSubtracted) {
  GripperContactConfig c = enforce_config();
  // Threshold above the bias but below bias+contact jump.
  c.torque_threshold = 0.5;
  GripperContactDetector d;
  d.configure(c);
  // Baseline torque 0.4 (offset/friction), contact adds +0.3 -> delta 0.3 < 0.5
  for (int i = 0; i < 5; ++i) {
    d.update(0.4, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 20; ++i) {
    d.update(0.7, kOpen, 0.0, kClose, kDt);
  }
  EXPECT_NE(d.state(), GripperContactState::kContactLatched);
  EXPECT_NEAR(d.torque_bias(), 0.4, 1e-6);
  EXPECT_NEAR(d.contact_delta(), 0.3, 1e-3);

  // Bigger jump 0.6 above bias -> contact
  GripperContactDetector d2;
  d2.configure(c);
  for (int i = 0; i < 5; ++i) {
    d2.update(0.4, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 20; ++i) {
    d2.update(1.0, kOpen, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d2.state(), GripperContactState::kContactLatched);
}

// 7. filter behaves like EMA (alpha=0.5)
TEST(GripperContactDetector, FilterWorks) {
  GripperContactDetector d;
  d.configure(enforce_config());
  // First update initializes filter directly
  d.update(0.0, kOpen, 0.0, kClose, kDt);
  EXPECT_DOUBLE_EQ(d.filtered_torque(), 0.0);
  d.update(0.8, kOpen, 0.0, kClose, kDt);
  EXPECT_NEAR(d.filtered_torque(), 0.4, 1e-6);
  d.update(0.8, kOpen, 0.0, kClose, kDt);
  EXPECT_NEAR(d.filtered_torque(), 0.6, 1e-6);
}

// 8. hysteresis: below release threshold resets the counter
TEST(GripperContactDetector, HysteresisResetsCounter) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  // 3 cycles over trigger (not enough for confirm_cycles=5)
  for (int i = 0; i < 3; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_GT(d.contact_counter(), 0);
  // drop below release -> counter resets
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.contact_counter(), 0);
  EXPECT_NE(d.state(), GripperContactState::kContactLatched);
}

// 9. hard limit trips immediately (single cycle, no confirmation)
TEST(GripperContactDetector, HardLimitTripsImmediately) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  // One single cycle above the hard limit
  d.update(kHardLimit + 1.0, -0.20, 0.0, kClose, kDt);
  EXPECT_EQ(d.state(), GripperContactState::kHardLimit);
  EXPECT_TRUE(d.hard_limit());
  EXPECT_EQ(d.close_result(), GripperCloseResult::kHardTorqueLimit);
  // Enforce mode holds the position where the limit tripped
  EXPECT_DOUBLE_EQ(d.effective_command(kClose), -0.20);
}

// 10. timeout while closing without contact/goal
TEST(GripperContactDetector, CloseTimeout) {
  GripperContactConfig c = enforce_config();
  c.max_close_duration_sec = 0.5;
  GripperContactDetector d;
  d.configure(c);
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  // closing forever, torque below threshold
  for (int i = 0; i < 100; ++i) {
    d.update(kEmptyTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.state(), GripperContactState::kTimeout);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kTimeout);
}

// 11. after latch, close target no longer advances (hold is frozen)
TEST(GripperContactDetector, LatchFreezesTarget) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_TRUE(d.contact());
  const double held = d.hold_position();
  // keep commanding, position unchanged
  for (int i = 0; i < 50; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_DOUBLE_EQ(d.hold_position(), held);
  EXPECT_DOUBLE_EQ(d.effective_command(kClose), held);
}

// 12. reopen resets the latch and arms the next close
TEST(GripperContactDetector, ReopenResetsLatch) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_TRUE(d.contact());
  // Open command far beyond position
  d.update(kEmptyTorque, -0.20, 0.0, kOpen, kDt);
  EXPECT_FALSE(d.contact());
  EXPECT_EQ(d.state(), GripperContactState::kIdle);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kNone);
  // A new close can latch again
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_TRUE(d.contact());
  EXPECT_EQ(d.close_result(), GripperCloseResult::kContactDetected);
}

// position reached (no object) reports POSITION_REACHED
TEST(GripperContactDetector, PositionReachedNoObject) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  // position converges to the commanded close position
  for (int i = 0; i < 20; ++i) {
    d.update(kEmptyTorque, kClose, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.close_result(), GripperCloseResult::kPositionReached);
  EXPECT_EQ(d.state(), GripperContactState::kIdle);
}

// Regression: a real close trajectory starts with command == position and
// tracks within tolerance while moving. Neither the start transient nor the
// mid-trajectory pass may settle POSITION_REACHED; when the gripper then hits
// a block, the contact must be reported as CONTACT_DETECTED.
TEST(GripperContactDetector, TrajectoryStartTransientDoesNotPoisonResult) {
  GripperContactDetector d;
  d.configure(enforce_config());
  // Trajectory start: command == position == open (JTC holds the first point).
  d.update(kEmptyTorque, kOpen, 0.0, kOpen, kDt);
  // Close trajectory: command steps toward kClose while the position tracks
  // just inside the 0.005 tolerance, then physically stops at the block -0.20.
  double cmd = kOpen;
  double pos = kOpen;
  for (int i = 0; i < 50; ++i) {
    cmd += 0.005;
    pos = std::min(cmd - 0.004, -0.20);
    d.update(kEmptyTorque, pos, 0.0, cmd, kDt);
  }
  EXPECT_NEAR(cmd, kClose, 1e-9);
  // Position is now stuck at the block while the command keeps demanding the
  // close target -> sustained contact torque must latch CONTACT_DETECTED.
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_EQ(d.state(), GripperContactState::kContactLatched);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kContactDetected);
  EXPECT_DOUBLE_EQ(d.hold_position(), -0.20);
}

// sensor invalid while closing -> SENSOR_INVALID
TEST(GripperContactDetector, SensorInvalidWhileClosing) {
  GripperContactDetector d;
  d.configure(enforce_config());
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  d.update(0.0, kOpen, 0.0, kClose, kDt, /*sensor_valid=*/false);
  EXPECT_EQ(d.close_result(), GripperCloseResult::kSensorInvalid);
}

// monitor mode observes but never overrides the command
TEST(GripperContactDetector, MonitorNeverOverridesCommand) {
  GripperContactConfig c = enforce_config();
  c.mode = GripperContactMode::kMonitor;
  GripperContactDetector d;
  d.configure(c);
  for (int i = 0; i < 5; ++i) {
    d.update(kEmptyTorque, kOpen, 0.0, kClose, kDt);
  }
  for (int i = 0; i < 10; ++i) {
    d.update(kContactTorque, -0.20, 0.0, kClose, kDt);
  }
  EXPECT_TRUE(d.contact());
  EXPECT_EQ(d.close_result(), GripperCloseResult::kContactDetected);
  EXPECT_DOUBLE_EQ(d.effective_command(kClose), kClose);  // unchanged
  EXPECT_DOUBLE_EQ(d.kp_scale(), 1.0);
}

}  // namespace openarm_hardware

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
