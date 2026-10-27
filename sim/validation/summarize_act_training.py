#!/usr/bin/env python3
"""Turn persisted LeRobot ACT train/eval loss lines into audit JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


TRAIN_RE = re.compile(r"step:(?P<step>[0-9]+(?:\.[0-9]+)?K?)\b.*?\bloss:(?P<loss>[0-9.]+)")
EVAL_RE = re.compile(r"step (?P<step>[0-9]+): eval_loss=(?P<loss>[0-9.]+)")


def parse_step(value: str) -> int:
    return int(round(float(value[:-1]) * 1000)) if value.endswith("K") else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    train = [
        {"step": parse_step(match["step"]), "loss": float(match["loss"])}
        for match in TRAIN_RE.finditer(text)
    ]
    evaluation = [
        {"step": int(match["step"]), "loss": float(match["loss"])}
        for match in EVAL_RE.finditer(text)
    ]
    if not train or not evaluation:
        raise SystemExit("log is missing train or eval loss records")
    report = {
        "log": str(args.log.resolve()),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "train_split_episodes": 4,
        "eval_split_episodes": 1,
        "train": train,
        "eval": evaluation,
        "summary": {
            "first_train_step": train[0]["step"],
            "last_train_step": train[-1]["step"],
            "first_train_loss": train[0]["loss"],
            "last_train_loss": train[-1]["loss"],
            "train_loss_reduction_fraction": 1.0 - train[-1]["loss"] / train[0]["loss"],
            "first_eval_loss": evaluation[0]["loss"],
            "last_eval_loss": evaluation[-1]["loss"],
            "eval_loss_reduction_fraction": 1.0 - evaluation[-1]["loss"] / evaluation[0]["loss"],
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
