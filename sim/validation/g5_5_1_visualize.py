#!/usr/bin/env python3
"""Visualization utilities for G5.5.1 experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_trajectory_comparison(
    raw_30hz: np.ndarray,
    upsampled_250hz: np.ndarray,
    guarded_250hz: np.ndarray,
    output_path: Path,
    title: str = "Trajectory Comparison"
):
    """Plot 30Hz raw vs upsampled vs guarded trajectories."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    joint_names = [
        "R-J1", "R-J2", "R-J3", "R-J4", "R-J5", "R-J6", "R-J7", "R-Grip",
        "L-J1", "L-J2", "L-J3", "L-J4", "L-J5", "L-J6", "L-J7", "L-Grip",
    ]

    for idx, ax in enumerate(axes.flat):
        if idx >= 16:
            break

        # Plot raw 30Hz (sparsely sampled)
        raw_indices = np.linspace(0, len(upsampled_250hz) - 1, len(raw_30hz), dtype=int)
        ax.plot(raw_indices, raw_30hz[:, idx], 'o-', label='30Hz raw', alpha=0.7, markersize=4)

        # Plot upsampled 250Hz
        ax.plot(upsampled_250hz[:, idx], '-', label='250Hz upsampled', alpha=0.7, linewidth=1)

        # Plot guarded 250Hz
        ax.plot(guarded_250hz[:, idx], '-', label='250Hz guarded', alpha=0.7, linewidth=1)

        ax.set_title(joint_names[idx], fontsize=10)
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('Position (rad)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {output_path}")


def plot_velocity_analysis(
    hermite_250hz: np.ndarray,
    guard_limits: np.ndarray,
    output_path: Path,
    dt: float = 1.0 / 250.0
):
    """Plot velocity analysis with Guard limits."""
    velocity = np.diff(hermite_250hz, axis=0) / dt
    velocity = np.vstack([velocity, velocity[-1:]])

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle("Velocity Analysis vs Guard Limits", fontsize=16)

    joint_names = [
        "R-J1", "R-J2", "R-J3", "R-J4", "R-J5", "R-J6", "R-J7", "R-Grip",
        "L-J1", "L-J2", "L-J3", "L-J4", "L-J5", "L-J6", "L-J7", "L-Grip",
    ]

    for idx, ax in enumerate(axes.flat):
        if idx >= 16:
            break

        # Plot velocity
        ax.plot(np.abs(velocity[:, idx]), '-', label='|Velocity|', alpha=0.7, linewidth=1)

        # Plot limit
        ax.axhline(guard_limits[idx], color='r', linestyle='--', label=f'Limit ({guard_limits[idx]:.2f})')

        # Highlight violations
        violations = np.abs(velocity[:, idx]) > guard_limits[idx]
        if violations.any():
            violation_indices = np.where(violations)[0]
            ax.scatter(violation_indices, np.abs(velocity[violations, idx]),
                      color='red', s=10, alpha=0.5, label='Violations')

        ax.set_title(joint_names[idx], fontsize=10)
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('|Velocity| (rad/s)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved velocity plot: {output_path}")


def plot_acceleration_analysis(
    hermite_250hz: np.ndarray,
    guard_limits: np.ndarray,
    output_path: Path,
    dt: float = 1.0 / 250.0
):
    """Plot acceleration analysis with Guard limits."""
    velocity = np.diff(hermite_250hz, axis=0) / dt
    velocity = np.vstack([velocity, velocity[-1:]])

    acceleration = np.diff(velocity, axis=0) / dt
    acceleration = np.vstack([acceleration, acceleration[-1:]])

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle("Acceleration Analysis vs Guard Limits", fontsize=16)

    joint_names = [
        "R-J1", "R-J2", "R-J3", "R-J4", "R-J5", "R-J6", "R-J7", "R-Grip",
        "L-J1", "L-J2", "L-J3", "L-J4", "L-J5", "L-J6", "L-J7", "L-Grip",
    ]

    for idx, ax in enumerate(axes.flat):
        if idx >= 16:
            break

        # Plot acceleration
        ax.plot(np.abs(acceleration[:, idx]), '-', label='|Acceleration|', alpha=0.7, linewidth=1)

        # Plot limit
        ax.axhline(guard_limits[idx], color='r', linestyle='--', label=f'Limit ({guard_limits[idx]:.2f})')

        # Highlight violations
        violations = np.abs(acceleration[:, idx]) > guard_limits[idx]
        if violations.any():
            violation_indices = np.where(violations)[0]
            ax.scatter(violation_indices, np.abs(acceleration[violations, idx]),
                      color='red', s=10, alpha=0.5, label='Violations')

        ax.set_title(joint_names[idx], fontsize=10)
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('|Acceleration| (rad/s²)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved acceleration plot: {output_path}")


def plot_experiment_comparison(
    results_dir: Path,
    output_path: Path
):
    """Compare all experiments in a summary plot."""
    experiments = ["E0", "E1", "E1-ZOH", "E1-60Hz"]
    metrics = {
        "success": [],
        "max_lift": [],
        "hold_duration": [],
        "dual_contact_steps": [],
    }

    exp_labels = []

    # Load G5.5 E0 and E1 from original results
    g5_5_dir = results_dir.parent / "g5_5_executor_equivalence" / "seed_20260902_smoke"
    for exp in ["E0", "E1"]:
        result_path = g5_5_dir / f"{exp.lower()}_result.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
                exp_labels.append(f"G5.5 {exp}")
                metrics["success"].append(1.0 if data["task"]["success"] else 0.0)
                metrics["max_lift"].append(data["task"]["max_cube_lift_m"])
                metrics["hold_duration"].append(data["task"]["hold_duration_s"])
                metrics["dual_contact_steps"].append(data["task"]["dual_finger_contact_steps"])

    # Load G5.5.1 results
    g5_5_1_dir = results_dir
    for exp in ["E1-ZOH", "E1-60Hz"]:
        result_path = g5_5_1_dir / f"{exp.lower().replace('-', '_')}_result.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
                exp_labels.append(f"G5.5.1 {exp}")
                metrics["success"].append(1.0 if data["task"]["success"] else 0.0)
                metrics["max_lift"].append(data["task"]["max_cube_lift_m"])
                metrics["hold_duration"].append(data["task"]["hold_duration_s"])
                metrics["dual_contact_steps"].append(data["task"]["dual_finger_contact_steps"])

    if not exp_labels:
        print("No experiment results found for comparison")
        return

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("G5.5 vs G5.5.1 Experiment Comparison", fontsize=16)

    x_pos = np.arange(len(exp_labels))

    # Success
    axes[0, 0].bar(x_pos, metrics["success"], color=['green' if s > 0.5 else 'red' for s in metrics["success"]])
    axes[0, 0].set_title("Success")
    axes[0, 0].set_ylabel("Success (1=PASS, 0=FAIL)")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(exp_labels, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Max Lift
    axes[0, 1].bar(x_pos, metrics["max_lift"])
    axes[0, 1].axhline(0.05, color='r', linestyle='--', label='Target (5cm)')
    axes[0, 1].set_title("Max Lift")
    axes[0, 1].set_ylabel("Lift Height (m)")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(exp_labels, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Hold Duration
    axes[1, 0].bar(x_pos, metrics["hold_duration"])
    axes[1, 0].axhline(1.0, color='r', linestyle='--', label='Target (1s)')
    axes[1, 0].set_title("Hold Duration")
    axes[1, 0].set_ylabel("Duration (s)")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(exp_labels, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Dual Contact Steps
    axes[1, 1].bar(x_pos, metrics["dual_contact_steps"])
    axes[1, 1].set_title("Dual Contact Steps")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(exp_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def generate_all_plots(results_dir: Path):
    """Generate all visualization plots for G5.5.1."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nGenerating plots in: {plots_dir}")

    # Load guard limits
    import yaml
    params_path = Path(__file__).resolve().parents[2] / "configs/collection/openarm_cube_pick_expert_params.yaml"
    cfg = yaml.safe_load(params_path.read_text())
    guard_limits_vel = np.array(cfg["guard"]["max_velocity"])
    guard_limits_acc = np.array(cfg["guard"]["max_acceleration"])

    # Plot E1-ZOH trajectories
    e1_zoh_traj_path = results_dir / "e1_zoh_trajectories.npz"
    if e1_zoh_traj_path.exists():
        data = np.load(e1_zoh_traj_path)
        plot_trajectory_comparison(
            data["raw_30hz"],
            data["zoh_250hz"],
            data["guarded_250hz"],
            plots_dir / "e1_zoh_trajectories.png",
            title="E1-ZOH: 30Hz → ZOH → Guard"
        )

    # Plot E1-60Hz trajectories
    e1_60hz_traj_path = results_dir / "e1_60hz_trajectories.npz"
    if e1_60hz_traj_path.exists():
        data = np.load(e1_60hz_traj_path)
        plot_trajectory_comparison(
            data["raw_60hz"],
            data["hermite_250hz"],
            data["guarded_250hz"],
            plots_dir / "e1_60hz_trajectories.png",
            title="E1-60Hz: 60Hz → Hermite → Guard"
        )

        # Velocity and acceleration analysis for 60Hz
        plot_velocity_analysis(
            data["hermite_250hz"],
            guard_limits_vel,
            plots_dir / "e1_60hz_velocity_analysis.png"
        )
        plot_acceleration_analysis(
            data["hermite_250hz"],
            guard_limits_acc,
            plots_dir / "e1_60hz_acceleration_analysis.png"
        )

    # Compare all experiments
    plot_experiment_comparison(results_dir, plots_dir / "experiment_comparison.png")

    print(f"\nPlot generation complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        ROOT = Path(__file__).resolve().parents[2]
        results_dir = ROOT / "results/g5_5_1_action_representation_debug/seed_20260902"

    generate_all_plots(results_dir)
