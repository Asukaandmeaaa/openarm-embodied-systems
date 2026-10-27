// Copyright 2026
// Implementation of the KDL gravity-compensation dynamics wrapper.

#include "openarm_hardware/dynamics.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

namespace openarm_hardware {

Dynamics::Dynamics(std::string urdf_path, std::string start_link,
                   std::string end_link)
    : urdf_path_(std::move(urdf_path)),
      start_link_(std::move(start_link)),
      end_link_(std::move(end_link)) {}

bool Dynamics::Init() {
  std::ifstream file(urdf_path_);
  if (!file.is_open()) {
    std::cerr << "[Dynamics] Failed to open URDF file: " << urdf_path_
              << std::endl;
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  if (!urdf_model_.initString(buffer.str())) {
    std::cerr << "[Dynamics] Failed to parse URDF: " << urdf_path_
              << std::endl;
    return false;
  }
  if (!kdl_parser::treeFromUrdfModel(urdf_model_, kdl_tree_)) {
    std::cerr << "[Dynamics] Failed to extract KDL tree" << std::endl;
    return false;
  }
  if (!kdl_tree_.getChain(start_link_, end_link_, kdl_chain_)) {
    std::cerr << "[Dynamics] Failed to get KDL chain: " << start_link_
              << " -> " << end_link_ << std::endl;
    return false;
  }

  const auto n = kdl_chain_.getNrOfJoints();
  gravity_forces_.resize(n);
  gravity_forces_.data.setZero();
  solver_ = std::make_unique<KDL::ChainDynParam>(
      kdl_chain_, KDL::Vector(0.0, 0.0, -9.81));
  return true;
}

void Dynamics::GetGravity(const double* joint_position, double* gravity) {
  const auto n = kdl_chain_.getNrOfJoints();
  KDL::JntArray q(n);
  for (unsigned int i = 0; i < n; ++i) {
    q(i) = joint_position[i];
  }
  solver_->JntToGravity(q, gravity_forces_);
  for (unsigned int i = 0; i < n; ++i) {
    gravity[i] = gravity_forces_(i);
  }
}

}  // namespace openarm_hardware
