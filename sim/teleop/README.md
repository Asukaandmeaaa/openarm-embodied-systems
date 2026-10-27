# OpenArm cube-pick pilot collection

The P1 contract is frozen in `configs/collection/feature_contract.yaml`. The
only task prompt is `Pick up the orange cube with the right arm.` and the only
recorded cameras are ceiling and right wrist.

1. Run `sim/teleop/setup_collection_env.sh` once.
2. Run `sim/teleop/run_pilot.sh` from a graphical desktop.
3. Open the UI at http://127.0.0.1:8000 and keyboard teleop at
   http://127.0.0.1:8080. Click the keyboard page so it has focus.
4. Record 3–5 complete successful episodes. Mark collisions, drops, partial
   lifts, or holds shorter than one second as failed; cancel corrupted runs.
5. Validate with:

   `.venv/bin/python sim/validation/validate_pilot.py data/raw/openarm_cube_pick/pilot --min-success 3 --report results/p1_openarm_cube_pick/pilot_validation.json`

6. Replay at least one accepted episode through the same actuators with:

   `.venv/bin/python sim/validation/replay_episode.py data/raw/openarm_cube_pick/pilot/episodes/0`

Raw data is append-only. Do not repair or delete episodes in this directory;
express later exclusions in a cleaned manifest.
