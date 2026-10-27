import mujoco

from openarm_mujoco.v2 import JointResolver
from openarm_cube_into_black_box_expert.runtime import (
    assert_left_is_parked,
    park_left_arm,
    parked_left_command,
)
from openarm_cube_into_black_box_expert.collect import Collector
from openarm_cube_pick_expert import collect as legacy


SCENE = "official/openarm_mujoco/v2/cell/demo.xml"


def test_new_task_parks_left_arm_out_of_runtime_workspace():
    model = mujoco.MjModel.from_xml_path(SCENE)
    before = model.ngeom
    counts = park_left_arm(model)
    assert model.ngeom == before
    assert counts["left_geoms_disabled"] > 0
    assert counts["left_geoms_hidden"] == counts["left_geoms_disabled"]
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
        if "left" in name or "left" in body_name:
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
            assert model.geom_rgba[geom_id, 3] == 0


def test_parked_command_is_stable():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mapper = JointResolver(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    expected = parked_left_command(mapper, data.qpos)
    assert expected.shape == (8,)
    assert_left_is_parked(mapper, data.qpos, expected)
    changed = expected.copy()
    changed[0] += 0.01
    mapper.set_qpos(data.qpos, changed, "left")
    try:
        assert_left_is_parked(mapper, data.qpos, expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("left-arm motion was not detected")


def test_full_preflight_falls_back_and_records_selection(monkeypatch):
    collector = Collector.__new__(Collector)
    collector.cfg = {"full_candidate_preflight": True}
    monkeypatch.setattr(collector, "_ranked_candidate_indices", lambda index: [1, 2])
    calls = []

    def fake_run(self, index, record=True, candidate_index=None):
        calls.append((record, candidate_index))
        success = candidate_index == 2
        details = {
            "selected_candidate": {"index": candidate_index},
            "success_reason": "complete" if success else None,
            "fail_reason": None if success else "place_ik_failed",
            "final_cube_pose": [0.4, 0.0, 1.035] if success else None,
            "place_error_m": 0.0 if success else None,
            "place_height_error_m": 0.0 if success else None,
        }
        details.update(getattr(self, "_episode_context", {}))
        return success, details

    monkeypatch.setattr(legacy.Collector, "run_episode", fake_run)
    ok, details = collector.run_episode(3, record=True)

    assert ok
    assert calls == [(False, 1), (False, 2), (True, 2)]
    assert details["preflight_selected_candidate"] == 2
    assert [item["success"] for item in details["candidate_preflights"]] == [False, True]
