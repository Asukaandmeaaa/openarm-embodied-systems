# Publication contents

The public repository was prepared in a separate staging checkout. The active
robotics workspace was not moved, renamed, initialized as Git or modified.

## Included

- Main expert, collection, conversion, evaluation, execution and diagnostic source.
- Configuration and task contracts; simulation geometry and meshes.
- Root regression tests and small vision fixtures.
- Companion kinesthetic library and ROS recorder source, including its tests.
- Five main upstream source snapshots plus nested Dora nodes, with licenses.
- Current ROS tracked changes as patches and new source files as additions.
- Historical documentation labeled as such, and new root setup/reproduction guides.
- Python CPU environment lock, explicit offline Make targets and CI workflow.

`official/` snapshots are normal tracked files, not Git submodules. Nested `.git`
metadata is excluded, and upstream `.gitmodules` are renamed `UPSTREAM.gitmodules`
for reference. This preserves local changes without depending on unpublished forks.

## Excluded

- Robot control keystores and private keys, including `.control_ownership_keys/`.
- Virtual environments, caches, build/install/log overlays and runtime lock files.
- Full `data/`, `results/`, `outputs/`, model weights, recordings and collection runs.
- Live camera previews and workstation session-registration output.
- Backup source copies, personal editor/agent settings and nested Git metadata.

This is a source release with a tested CPU entry point. Excluding runtime artifacts
means some historical workflows still require external files. See REPRODUCIBILITY.md.
