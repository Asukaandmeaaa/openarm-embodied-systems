"""CLI for the Teaching Ready SE(3) pose search.

    python -m openarm_kinesthetic.ready_search                 # full search, writes the report
    python -m openarm_kinesthetic.ready_search --quick         # reduced grid, same code path
    python -m openarm_kinesthetic.ready_search --verify-determinism
    python -m openarm_kinesthetic.ready_search --render        # PNG per Top-N candidate
    python -m openarm_kinesthetic.ready_search --viewer        # interactive 1/2/3 comparison

Nothing here executes motion: the module compiles a MuJoCo scene, computes, and writes
a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_ready_search_config, quick_overrides, report_output_path
from .report import render_comparison_table, render_family_table, render_orientation_table
from .search import TeachingReadySearch, verify_determinism


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="reduced grid/refine profile for a fast run")
    parser.add_argument("--output", default=None, help="report file name under results/ready_search/")
    parser.add_argument("--no-write", action="store_true", help="print only, write no report file")
    parser.add_argument("--verify-determinism", action="store_true", help="run the search twice and compare Top-N")
    parser.add_argument("--render", action="store_true", help="write one PNG per Top-N candidate")
    parser.add_argument("--render-only", default=None, help="render an existing report file instead of searching")
    parser.add_argument("--viewer", action="store_true", help="open the interactive Top-N comparison viewer")
    parser.add_argument("--viewer-only", default=None, help="view an existing report file instead of searching")
    parser.add_argument("--json", action="store_true", help="also print the report JSON on stdout")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_ready_search_config(quick_overrides() if args.quick else None)

    if args.render_only or args.viewer_only:
        from .visualize import load_report
        from .report import render_comparison_table as _table

        path = Path(args.render_only or args.viewer_only)
        report: dict[str, Any] = load_report(path)
        print(_table(report))
        if args.render_only:
            from .visualize import render_candidates

            for written in render_candidates(report, config=config):
                print(f"[ready-search] wrote {written}")
        if args.viewer_only:
            from .visualize import view_candidates

            view_candidates(report, config=config)
        return

    search = TeachingReadySearch(config)
    report = search.run()

    if args.verify_determinism:
        determinism = verify_determinism(config)
        report["determinism"] = determinism
        report["acceptance"]["determinism_check"] = (
            "verified in this run" if determinism["deterministic"] else "FAILED in this run"
        )
        report["acceptance"]["checks"]["top_n_reproducible"] = bool(determinism["deterministic"])
        if not determinism["deterministic"]:
            report["acceptance"].setdefault("failed_checks", []).append("top_n_reproducible")
            report["acceptance"]["status"] = "FAIL"

    print(render_comparison_table(report))
    print()
    print(render_family_table(report))
    print()
    print(render_orientation_table(report))
    print()

    output: Path | None = None
    if not args.no_write:
        output = report_output_path(config, args.output)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ready-search] report: {output}")

    if args.json:
        print(json.dumps(report["acceptance"], indent=2))

    if args.render:
        from .visualize import render_candidates

        for written in render_candidates(report, config=config):
            print(f"[ready-search] wrote {written}")

    if args.viewer:
        from .visualize import view_candidates

        view_candidates(report, config=config)

    print(
        f"[ready-search] acceptance {report['acceptance']['status']}  "
        f"failed={report['acceptance']['failed_checks']}  "
        f"candidates={report['counts']['candidates_valid']}  "
        f"planned={report['counts']['candidates_planned']}"
    )
    if report["acceptance"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
