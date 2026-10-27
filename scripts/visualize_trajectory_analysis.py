#!/usr/bin/env python3
"""Visualize trajectory analysis results with stage-by-stage deviation plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_dir: Path) -> list[dict[str, Any]]:
    """Load all trajectory analysis results from directory."""
    results = []
    for json_file in sorted(result_dir.glob("trajectory_analysis_seed*.json")):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


def plot_stage_deviations(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot mean and max deviations for each stage across seeds."""
    stages = ["upsampler", "lowpass", "guard"]
    stage_labels = ["Hermite\nUpsampler", "Biquad\nLowpass", "Action\nGuard"]
    seeds = [r["seed"] for r in results]
    n_seeds = len(seeds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric_idx, metric in enumerate(["mean", "max"]):
        ax = axes[metric_idx]

        x = np.arange(len(stages))
        width = 0.8 / n_seeds

        for seed_idx, result in enumerate(results):
            values = [
                result["stage_analysis"][stage]["deviation"][metric]
                for stage in stages
            ]
            offset = (seed_idx - n_seeds / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=f'Seed {result["seed"]}',
                alpha=0.8,
            )

        ax.set_ylabel(f"{metric.capitalize()} Deviation (rad)", fontsize=11)
        ax.set_title(f"{metric.capitalize()} Trajectory Deviation by Stage", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved deviation plot: {output_path}")


def plot_guard_clamps(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot ActionGuard clamp statistics across seeds."""
    clamp_types = ["velocity_clamps", "acceleration_clamps", "step_clamps", "ctrlrange_clamps"]
    clamp_labels = ["Velocity", "Acceleration", "Step", "Ctrl Range"]

    fig, ax = plt.subplots(figsize=(10, 6))

    seeds = [r["seed"] for r in results]
    x = np.arange(len(clamp_types))
    width = 0.8 / len(seeds)

    for seed_idx, result in enumerate(results):
        clamps = result["stage_analysis"]["guard"]["clamps"]
        values = [clamps[clamp_type] for clamp_type in clamp_types]
        offset = (seed_idx - len(seeds) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=f'Seed {result["seed"]}',
            alpha=0.8,
        )

    ax.set_ylabel("Clamp Count", fontsize=11)
    ax.set_title("ActionGuard Clamp Statistics", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clamp_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved clamp plot: {output_path}")


def plot_per_axis_deviation(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot per-axis max deviation for each stage."""
    stages = ["upsampler", "lowpass", "guard"]
    stage_labels = ["Hermite Upsampler", "Biquad Lowpass", "Action Guard"]

    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 10), sharex=True)

    axis_labels = [f"R{i}" for i in range(8)] + [f"L{i}" for i in range(8)]

    for stage_idx, stage in enumerate(stages):
        ax = axes[stage_idx]

        for result in results:
            per_axis = result["stage_analysis"][stage]["deviation"]["per_axis_max"]
            ax.plot(
                per_axis,
                marker="o",
                label=f'Seed {result["seed"]}',
                alpha=0.7,
            )

        ax.set_ylabel("Max Deviation (rad)", fontsize=10)
        ax.set_title(f"{stage_labels[stage_idx]} - Per-Axis Deviation", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Joint Axis", fontsize=11)
    axes[-1].set_xticks(range(16))
    axes[-1].set_xticklabels(axis_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-axis plot: {output_path}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print text summary of results."""
    print("\n" + "=" * 80)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 80)

    stages = ["upsampler", "lowpass", "guard"]

    for stage in stages:
        print(f"\n{stage.upper()}:")
        mean_devs = [r["stage_analysis"][stage]["deviation"]["mean"] for r in results]
        max_devs = [r["stage_analysis"][stage]["deviation"]["max"] for r in results]

        print(f"  Mean deviation: {np.mean(mean_devs):.6f} ± {np.std(mean_devs):.6f} rad")
        print(f"  Max deviation:  {np.mean(max_devs):.6f} ± {np.std(max_devs):.6f} rad")

        if stage == "guard":
            print(f"  Clamps (total across seeds):")
            total_clamps = {}
            for r in results:
                clamps = r["stage_analysis"]["guard"]["clamps"]
                for k, v in clamps.items():
                    total_clamps[k] = total_clamps.get(k, 0) + v

            for clamp_type, count in sorted(total_clamps.items()):
                if count > 0:
                    print(f"    {clamp_type}: {count}")

    print("\n" + "=" * 80)
    print("TRAJECTORY SMOOTHING RESPONSIBILITY")
    print("=" * 80)
    if results:
        boundary = results[0]["trajectory_smoothing_responsibility"]
        print(f"Primary smoother: {boundary['primary_smoother']}")
        print(f"Rationale: {boundary['rationale']}")
        print(f"Upsampler role: {boundary['upsampler_role']}")
        print(f"Guard role: {boundary['guard_role']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path, help="Directory containing trajectory analysis JSON files")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots (default: same as result_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.result_dir)
    if not results:
        print(f"No trajectory analysis results found in {args.result_dir}")
        return

    print(f"Loaded {len(results)} results from seeds: {[r['seed'] for r in results]}")

    print_summary(results)

    plot_stage_deviations(results, output_dir / "stage_deviations.png")
    plot_guard_clamps(results, output_dir / "guard_clamps.png")
    plot_per_axis_deviation(results, output_dir / "per_axis_deviation.png")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
