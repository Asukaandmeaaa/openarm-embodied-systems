"""Compile the independent expert scene and advance physics; no renderer or I/O."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.act_sim_mix.sim_expert.world import SimWorld, load_config
from experiments.act_sim_mix.contracts import digest
from experiments.act_sim_mix.sim_expert.world import SCENE


def main():
    world = SimWorld(20261028, load_config())
    try:
        world.hold_for(.1, 'SMOKE')
        state = world.state16()
        if state.shape != (16,) or not np.isfinite(state).all():
            raise RuntimeError('invalid 16-dimensional simulated feedback')
        assert not hasattr(world, '_renderer')
        print(json.dumps(dict(hardware_accessed=False,rendered=False,
            simulation_time_s=float(world.data.time),samples=len(world.samples),
            state_shape=list(state.shape),scene_sha256=digest(SCENE)),indent=2))
    finally:
        world.close()


if __name__ == '__main__':
    main()
