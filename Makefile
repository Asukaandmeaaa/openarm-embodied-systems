.PHONY: test smoke test-sim-mix smoke-sim-mix

test:
	uv run --frozen python -m pytest -q tests/test_expert_contracts.py tests/test_control_ownership.py tests/test_cube_into_black_box_contract.py tests/test_act_trial_speed_contract.py

smoke:
	uv run --frozen python tools/smoke_sim.py

test-sim-mix:
	uv run --frozen --group sim-mix python -m pytest -q experiments/act_sim_mix/tests

smoke-sim-mix:
	uv run --frozen --group sim-mix python tools/smoke_act_sim_mix.py
