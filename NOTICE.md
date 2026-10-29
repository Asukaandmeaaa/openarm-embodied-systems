# Notices

This repository contains original integration, validation, experiment, and runtime-safety code by Asukaandmeaaa together with attributed upstream source snapshots.

The `official/` tree contains material derived from the following upstream projects:

- Enactic [openarm_mujoco](https://github.com/enactic/openarm_mujoco)
- Enactic [openarm_dataset](https://github.com/enactic/openarm_dataset)
- Enactic [dora-openarm-keyboard](https://github.com/enactic/dora-openarm-keyboard)
- Enactic [dora-openarm-data-collection](https://github.com/enactic/dora-openarm-data-collection)
- Enactic [dora-openarm-evaluation](https://github.com/enactic/dora-openarm-evaluation)

The original Apache-2.0 license files remain in their respective directories. These source snapshots include local integration modifications.

Additional attributed material includes the nested Dora nodes, the openarm_can
vendor tree, the companion MuJoCo snapshot, and ROS/OpenArm patches and recovery
source. Original copyright headers and available licenses are retained.

`experiments/act_sim_mix/sim_reference.py` retains the Hermite upsampler and
biquad lowpass classes from Enactic's dora-openarm-evaluation under Apache-2.0.
It contains only offline reference mathematics. Source hashes and port details
are recorded in that experiment's `config/` directory, with a retained license
at `assets/REFERENCE_LICENSE`. Its MJCF uses the existing attributed
`official/openarm_mujoco/` assets rather than redistributing another asset copy.
