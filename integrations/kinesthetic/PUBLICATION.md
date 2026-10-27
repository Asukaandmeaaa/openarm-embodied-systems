# Teaching source snapshot

This directory preserves the source, configuration examples, ROS package, tests,
and simulation resources from the companion `openarm_kinesthetic_ws` workspace.
Its original layout and historical README are retained; it is not wired into the
root CPU quickstart. The original project expected a sibling `openarm_ws` and
`openarm_vla_ws`; relocate or parameterize those paths before running its commands.

No ROS install/build trees, demonstrations, live credentials or runtime reports
are included. `vendor/openarm_mujoco` is the companion's own frozen snapshot; it
can differ from the root `official/openarm_mujoco` tree and must not be silently
substituted for it. Preserve its LICENSE and vendor manifest when updating.
