# Reproduction status — source release 0.1.0

## Tested CPU baseline

- Linux x86_64; CPython 3.12.
- `uv sync --frozen` recreates the isolated dependency environment in `.venv`.
- Exact direct and transitive versions and artifact hashes are in `uv.lock`.
- `make test` runs four explicitly selected test modules: expert contracts,
  control ownership, placement contracts and speed configuration.
- `make smoke` loads the shipped MuJoCo scene and advances 100 physics steps,
  without rendering, ROS, CAN, cameras, checkpoints or external workspaces.

The baseline uses NumPy 2.2.6, SciPy 1.18.0, MuJoCo 3.8.1, PyArrow 25.0.0,
pandas 2.3.3, Pillow 12.2.0, PyYAML 6.0.3 and pytest 9.1.1. These versions
were taken from the original working environment and installed afresh for validation.
The vendored `openarm_mujoco` and `openarm_dataset` Python sources are added to
pytest's import path by `pyproject.toml`; no installed workstation copy is needed.

```bash
uv sync --frozen
make test
make smoke
```

Python dependencies can also be installed with `requirements/sim.txt`, but the
`uv.lock` route is the tested baseline. Source packages rely on the checkout's
configurations and assets; wheel-only installation is not a supported deployment.

## What is not covered by that baseline

| Workflow | Included inputs | Additional prerequisites / status |
| --- | --- | --- |
| ACT / LeRobot training | Conversion, validation and training source; configuration examples | Python 3.12, LeRobot 0.6.1, suitable PyTorch/CUDA, datasets, split manifests and model assets; optional requirements have not been clean-room validated |
| Gamepad / kinesthetic teaching | `integrations/kinesthetic/`, ROS recorder, scenes and vendor snapshot | External ROS/OpenArm workspace, package builds and adapted paths; root supervisor still uses original workspace locations |
| Real vision collection | Collection source, small regression fixtures, reference calibration example | RealSense/OpenCV/ROS environment, new camera calibration and local registration |
| ROS hardware | Pinned upstreams in `ros2.repos`, local patches/additions, separate recovery source | Reconstruct ROS sources, install system dependencies, build and commission independently; not a validated physical release |
| Synchronized ACT guard | Original Ruckig integration source | Exact 0.12.2 CPython 3.12 Linux binary and SHA256 expected by `load_ruckig`; binary is deliberately not redistributed or bypassed |
| Closed-transit ACT profile | Profile and model verification logic | Pinned collected `robot.urdf`, current MoveIt limits, recordings and checkpoint; excluded artifacts are still required |

Running `pytest tests` is **not** the documented portable test command. Many
historical tests intentionally read recordings, generated reports, ROS installations
or sibling workspaces. The initial three-module audit passed 136 tests and failed
three cases because the pinned Ruckig binary and collected robot model were absent.
That workstation-assisted audit is not equivalent to the clean CPU baseline.
No safety gate or model hash check was loosened to make the release tests pass.

## Data and artifact contract

The simulation examples use a 16-axis vector ordered as right arm joints 1–7,
right gripper, left arm joints 1–7, left gripper. Camera names, resolution, timing,
gripper conventions and action semantics must be taken from the selected task's
feature contract; they vary between real and simulated experiments.

Full datasets and checkpoints have not been published in this release. Historical
paths in training recipes do not denote downloadable artifacts. Reproducing a
reported training result additionally requires the original episode list, split,
conversion revision, model configuration, normalizers, checkpoint and checksums.
No download URL or dataset license is invented here.

Large artifacts should be published separately with immutable versions, checksums,
data/model cards and a license. Small synthetic fixtures remain in Git even when
they use a binary data format. Do not blanket-ignore all `.npy` or Parquet inputs.

## Configuration and historical records

Training paths under the main workspace were made repository-relative. Most other
historical paths remain intact as provenance, including machine-specific shell and
ROS launch scripts. They are not supported quickstart defaults. `.calib.example`
files are reference examples only; configure calibration for the actual device.

Historical documentation may refer to logs or recordings intentionally not shipped.
Use the root README as the current entry point. Configuration changes can invalidate
saved hashes and physical validation; do not replace missing assets with unchecked
fallbacks or execute stale operational commands from archived notes.

## Release validation record

The staged Git tree was extracted into a fresh `/tmp` directory with no sibling
robotics workspaces. `uv sync --frozen`, `make test` (108 passed), and `make smoke`
all completed successfully. See [release-validation.json](release-validation.json).
This verifies only the explicit CPU baseline above.
