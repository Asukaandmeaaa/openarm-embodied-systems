# Upstream sources

The `official/` directories are vendored source snapshots. They are not claimed as original work. The snapshots below were taken from the listed upstream repositories; local integration changes remain in the snapshot and are covered by the surrounding tests and documentation.

| Local path | Upstream repository | Source commit | Local changes present when staged |
| --- | --- | --- | --- |
| `official/dora-openarm-data-collection` | https://github.com/enactic/dora-openarm-data-collection.git | `dd1bab22a92b2828c598a5f42b89572f651b60e1` | yes |
| `official/dora-openarm-evaluation` | https://github.com/enactic/dora-openarm-evaluation.git | `d15939afbe5ae30aa0766b3866e58f90e61528b0` | yes |
| `official/dora-openarm-keyboard` | https://github.com/enactic/dora-openarm-keyboard.git | `2ba2ed9edef399b7457e9f62ec5a385d77e4eaf4` | yes |
| `official/openarm_dataset` | https://github.com/enactic/openarm_dataset.git | `a69fe37384e26bb5c3059de4a52cd6e871727011` | yes |
| `official/openarm_mujoco` | https://github.com/enactic/openarm_mujoco.git | `a8c979629f2591ad035d99d338ce114969e6cddc` | yes |

Each upstream directory retains its original license file. When a future release updates one of these snapshots, update this table and record the corresponding upstream commit in the release notes.

## Nested components and ROS changes

[`upstream-revisions.json`](upstream-revisions.json) records nested Dora node
origins, commits, tracked changes and untracked-file names at capture time.
These source snapshots contain local modifications: the listed commit is the
upstream base, not a claim that the files are identical to that commit.

The companion teaching snapshot is in `integrations/kinesthetic`; its local Git
revision is recorded even though it has no configured public origin. Its separate
MuJoCo vendor manifest is retained. `integrations/ros2/sources.json` records the
external ROS bases, and that directory preserves patches and source additions.
`vision_grasp_baseline/recovery_source` is a distinct historical recovery snapshot.

Vendored openarm_can source under the Dora evaluation tree retains its own license.
All original copyright/license headers are retained. The root project metadata is
not a claim of authorship over upstream implementations.
