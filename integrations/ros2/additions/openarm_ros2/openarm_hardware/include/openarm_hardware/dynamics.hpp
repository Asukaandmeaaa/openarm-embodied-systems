// Copyright 2026
// Gravity compensation dynamics wrapper (KDL-based).
// Computes the gravity torque feedforward G(q) for one arm chain of the
// OpenArm URDF so that the MIT position controller no longer has to carry
// the full gravity load (reduces steady-state droop at high-load postures).

#pragma once

#include <memory>
#include <string>

#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

namespace openarm_hardware {

/**
 * @brief KDL dynamics wrapper computing gravity torques G(q) for a single
 * arm chain of the OpenArm URDF.
 *
 * The chain must contain exactly the arm joints (joint1..joint7), e.g.
 *   base: openarm_left_base_link   tip: openarm_left_ee_base_link
 */
class Dynamics {
 public:
  Dynamics(std::string urdf_path, std::string start_link,
           std::string end_link);

  /** Builds the KDL tree/chain and the ChainDynParam solver. */
  bool Init();

  /** Number of joints in the configured chain (must match ARM_DOF). */
  std::size_t JointCount() const { return kdl_chain_.getNrOfJoints(); }

  /**
   * @brief Fills gravity[0..n-1] with the joint gravity torques G(q) [Nm].
   * @param joint_position input joint angles [rad], order = chain joint order
   * @param gravity        output gravity torques [Nm]
   */
  void GetGravity(const double* joint_position, double* gravity);

 private:
  std::string urdf_path_;
  std::string start_link_;
  std::string end_link_;

  urdf::Model urdf_model_;
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_;
  KDL::JntArray gravity_forces_;
  std::unique_ptr<KDL::ChainDynParam> solver_;
};

}  // namespace openarm_hardware
