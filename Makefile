.PHONY: test smoke

test:
	uv run --frozen python -m pytest -q tests/test_expert_contracts.py tests/test_control_ownership.py tests/test_cube_into_black_box_contract.py tests/test_act_trial_speed_contract.py

smoke:
	uv run --frozen python tools/smoke_sim.py
