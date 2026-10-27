#!/usr/bin/env python3
"""Run LeRobot training while persisting its console metrics to a log file."""

from __future__ import annotations

import os
from pathlib import Path

from lerobot.scripts import lerobot_train


LOG_PATH = Path(
    os.environ.get(
        "OPENARM_TRAIN_LOG",
        "results/act_training/act_openarm_cube_pick_g5_500.log",
    )
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_base_init_logging = lerobot_train.init_logging


def _init_logging_with_file(*, accelerator=None, **kwargs):
    return _base_init_logging(
        log_file=LOG_PATH,
        accelerator=accelerator,
        **kwargs,
    )


lerobot_train.init_logging = _init_logging_with_file


if __name__ == "__main__":
    lerobot_train.main()
