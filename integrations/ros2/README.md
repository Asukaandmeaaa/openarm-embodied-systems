# ROS source provenance

`../../ros2.repos` pins the upstream bases. `openarm_ros2.patch` and
`openarm_description.patch` preserve tracked workstation changes relative to those
bases; `additions/` preserves new source files (backup copies excluded).
The separate `vision_grasp_baseline/recovery_source/` tree is a recovery snapshot,
not necessarily identical to the current ROS workspace.

For source reconstruction only, import the `.repos` file into a new ROS workspace,
apply each patch from its corresponding repository root with `git apply`, and copy
that repository's additions over it. Review local paths and ROS package dependencies
before compiling. No build, launch, calibration or hardware verification was performed
for this source release. These changes must not be represented as a clean upstream checkout.
