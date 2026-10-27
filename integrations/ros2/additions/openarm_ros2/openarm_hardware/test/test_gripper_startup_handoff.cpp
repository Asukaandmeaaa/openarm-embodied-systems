// Startup cache handoff regression. No on_init/on_configure/on_activate calls:
// no SDK instance, ROS node, CAN socket, motor enable, or physical commands.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "openarm_hardware/openarm_simple_hardware.hpp"

namespace openarm_hardware {

class OpenArmHWStartupTest : public ::testing::Test {
 protected:
  OpenArmHW hw;

  void SetUp() override { configure("right_", "pinch_gripper"); }

  void configure(const std::string& side, const std::string& type, bool hand = true) {
    hw.hand_ = hand;
    hw.arm_prefix_ = side;
    hw.ee_type_ = type;
    hw.generate_joint_names();
    const auto size = hw.joint_names_.size();
    hw.pos_states_.assign(size, 0.0);
    hw.vel_states_.assign(size, 0.0);
    hw.tau_states_.assign(size, 0.0);
    hw.pos_commands_.assign(size, 0.0);
    hw.vel_commands_.assign(size, 0.0);
    hw.tau_commands_.assign(size, 0.0);
  }

  bool feedback(double position, double velocity = 0.0, double torque = 0.0) {
    return hw.store_gripper_feedback(position, velocity, torque);
  }
  void seed(double q) { hw.seed_gripper_hold(q); }
  void configure_handoff() { seed(hw.pos_states_[7]); }
  void home_handoff() { seed(hw.startup_gripper_target()); }
  void closed_collection() { hw.startup_gripper_closed_ = true; }
  bool parse(const std::string& flag, const std::string& side = "right_",
             const std::string& type = "pinch_gripper", bool hand = true) {
    hardware_interface::HardwareInfo info;
    info.hardware_parameters = {{"arm_prefix", side}, {"ee_type", type},
                                {"hand", hand ? "true" : "false"}};
    if (!flag.empty()) info.hardware_parameters["startup_gripper_closed"] = flag;
    return hw.parse_config(info);
  }
  double target() const { return hw.pos_commands_[7]; }
  double state() const { return hw.pos_states_[7]; }
  double speed() const { return hw.vel_states_[7]; }
  double torque() const { return hw.tau_states_[7]; }
  double motor_target() { return hw.joint_to_motor_radians(target()); }
  std::vector<double> arm_targets() const {
    return {hw.pos_commands_.begin(), hw.pos_commands_.begin() + 7};
  }
  void stale_feedforward() {
    hw.vel_commands_[7] = .7;
    hw.tau_commands_[7] = 1.0;
  }
  bool feedforward_zero() const {
    return hw.vel_commands_[7] == 0. && hw.tau_commands_[7] == 0.;
  }
};

TEST_F(OpenArmHWStartupTest, ConfigureUsesSDKFeedbackInsteadOfZeroInitializedCache) {
  ASSERT_DOUBLE_EQ(target(), 0.);
  ASSERT_TRUE(feedback(-.044823, -.007326, .178));
  configure_handoff();
  EXPECT_DOUBLE_EQ(state(), -.044823);
  EXPECT_DOUBLE_EQ(target(), -.044823);
  EXPECT_DOUBLE_EQ(speed(), -.007326);
  EXPECT_DOUBLE_EQ(torque(), .178);
}

TEST_F(OpenArmHWStartupTest, RightHomeTargetPersistsUntilControllerClaimsInterfaces) {
  ASSERT_TRUE(feedback(-.044823));
  configure_handoff();
  const auto arms_before = arm_targets();
  stale_feedforward();
  home_handoff();
  ASSERT_TRUE(feedback(-.332, -.007326, .12));
  EXPECT_DOUBLE_EQ(state(), -.332);  // Keep real feedback; do not invent arrival.
  EXPECT_DOUBLE_EQ(target(), -.35);
  EXPECT_TRUE(feedforward_zero());
  EXPECT_EQ(arm_targets(), arms_before);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_DOUBLE_EQ(motor_target(), -.35);
  }
}

TEST_F(OpenArmHWStartupTest, ExportedControllerInterfacesSeeOpenStateAndOpenReference) {
  ASSERT_TRUE(feedback(-.04));
  configure_handoff();
  home_handoff();
  ASSERT_TRUE(feedback(-.34));
  auto states = hw.export_state_interfaces();
  auto commands = hw.export_command_interfaces();
  const std::size_t gripper_position = 7 * 3;
  EXPECT_DOUBLE_EQ(states[gripper_position].get_optional().value(), -.34);
  EXPECT_DOUBLE_EQ(commands[gripper_position].get_optional().value(), -.35);
  // Both supported controller initializations (measured or previous command)
  // stay open; neither exported value is the old 0-rad closed default.
  ASSERT_TRUE(commands[gripper_position].set_value(states[gripper_position].get_optional().value()));
  EXPECT_DOUBLE_EQ(motor_target(), -.34);
}

TEST_F(OpenArmHWStartupTest, UncontrolledLeftPinchGripperKeepsInitialMeasuredHold) {
  configure("left_", "pinch_gripper");
  ASSERT_TRUE(feedback(.12));
  configure_handoff();
  home_handoff();
  EXPECT_DOUBLE_EQ(target(), .12);
  EXPECT_DOUBLE_EQ(motor_target(), .12);
}

TEST_F(OpenArmHWStartupTest, CollectionClosedTargetPersistsWithoutFabricatingFeedback) {
  ASSERT_TRUE(feedback(-.35));
  configure_handoff();
  closed_collection();
  home_handoff();
  EXPECT_DOUBLE_EQ(target(), -.085);
  EXPECT_DOUBLE_EQ(state(), -.35);
  for (int i = 0; i < 1000; ++i) EXPECT_DOUBLE_EQ(motor_target(), -.085);
}

TEST_F(OpenArmHWStartupTest, ClosedCollectionFlagNeverChangesUncontrolledLeftTarget) {
  configure("left_", "pinch_gripper");
  ASSERT_TRUE(feedback(.12));configure_handoff();closed_collection();home_handoff();
  EXPECT_DOUBLE_EQ(target(), .12);
}

TEST_F(OpenArmHWStartupTest, ClosedCollectionRequiresExplicitStrictCompatibleConfiguration) {
  ASSERT_TRUE(parse(""));home_handoff();EXPECT_DOUBLE_EQ(target(), -.35);
  ASSERT_TRUE(parse("false"));home_handoff();EXPECT_DOUBLE_EQ(target(), -.35);
  ASSERT_TRUE(parse("true"));home_handoff();EXPECT_DOUBLE_EQ(target(), -.085);
  EXPECT_FALSE(parse("tru"));
  EXPECT_FALSE(parse("true", "left_"));
  EXPECT_FALSE(parse("true", "right_", "parallel_link"));
  EXPECT_FALSE(parse("true", "right_", "pinch_gripper", false));
}

TEST_F(OpenArmHWStartupTest, ParallelGripperRetainsMeterToMotorConversion) {
  configure("right_", "parallel_link");
  ASSERT_TRUE(feedback(-.5236, -.1, .2));
  configure_handoff();
  EXPECT_NEAR(state(), .022, 1e-12);
  EXPECT_NEAR(speed(), .1*.044/1.0472, 1e-12);
  home_handoff();
  EXPECT_DOUBLE_EQ(target(), .044);
  EXPECT_NEAR(motor_target(), -1.0472, 1e-12);
}

TEST_F(OpenArmHWStartupTest, NonfiniteFeedbackCannotOverwriteValidState) {
  ASSERT_TRUE(feedback(-.34));
  EXPECT_FALSE(feedback(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(feedback(-.34, std::numeric_limits<double>::infinity()));
  EXPECT_DOUBLE_EQ(state(), -.34);
  // Existing contact diagnostics must still receive invalid torque as-is.
  EXPECT_TRUE(feedback(-.34, 0., std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(std::isnan(torque()));
}

TEST_F(OpenArmHWStartupTest, HandDisabledDoesNotAccessOrAddEighthAxis) {
  configure("right_", "pinch_gripper", false);
  const auto before = arm_targets();
  seed(-.35);
  EXPECT_EQ(hw.export_command_interfaces().size(), 21u);
  EXPECT_EQ(arm_targets(), before);
}

}  // namespace openarm_hardware
