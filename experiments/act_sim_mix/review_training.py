"""Read saved validation summaries; never runs inference or selects/deploys a model."""
import argparse
import json
from pathlib import Path

from .metrics import validation_brief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.run.glob("summary_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    if not files:
        parser.error("no completed validation summaries in run")
    if args.output and args.output.exists():
        parser.error("output exists; choose a new report path")
    report = {"run": str(args.run.resolve()), "checkpoints": [], "hardware_accessed": False,
              "selection": "Validation diagnostics only. Review closed-loop results before selection; final test remains held out."}
    print("step\tdataset\tloss\tarm_MAE_deg_h0\tgripper_MAE_rad_h0\tpregrasp_open_miss_fraction")
    for path in files:
        raw = json.loads(path.read_text())
        summaries = {name: validation_brief(row) for name, row in raw["evaluation"].items()}
        report["checkpoints"].append({"step": raw["step"], "summary": str(path.resolve()), "evaluation": summaries})
        for name, brief in summaries.items():
            h0, opening = brief["h0"] or {}, brief["failure_to_open"] or {}
            values = (raw["step"], name, brief["loss"], h0.get("arm_mae_deg"), h0.get("gripper_mae_rad"), opening.get("fraction"))
            print("\t".join("NA" if v is None else str(v) for v in values))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
