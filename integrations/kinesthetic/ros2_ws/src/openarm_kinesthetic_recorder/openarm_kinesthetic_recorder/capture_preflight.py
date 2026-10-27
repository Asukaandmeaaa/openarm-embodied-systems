"""Truthful preflight metadata for attended capture; no hardware operations."""


def capture_preflight(mode, plan_only_passed, dry_replay_passed):
    if mode not in ("prior_verified", "inline"):
        raise ValueError("capture validation mode must be prior_verified or inline")
    for name, value in (("plan_only_passed", plan_only_passed), ("dry_replay_passed", dry_replay_passed)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        if mode == "prior_verified" and value is not True:
            raise ValueError(f"{name} must be true for prior_verified capture")
        if mode == "inline" and value is not False:
            raise ValueError(f"{name} must be false for inline capture; do not claim separate preflight")
    return {"capture_validation_mode": mode,
            "plan_only_passed": plan_only_passed,
            "dry_replay_passed": dry_replay_passed}


def batch_suffix(paths, start_number):
    if type(start_number) is not int or not 1 <= start_number <= len(paths):
        raise ValueError(f"trajectory_start_number must be 1..{len(paths)}")
    return paths[start_number-1:]
