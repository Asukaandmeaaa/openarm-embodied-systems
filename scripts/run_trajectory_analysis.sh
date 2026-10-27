#!/usr/bin/env bash
# Run expert vs ACT trajectory analysis for multiple seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:-$ROOT_DIR/outputs/act_openarm_cube_pick_smoke_g5_500/checkpoints/last}"
OUTPUT_DIR="${2:-$ROOT_DIR/results/trajectory_analysis_$(date +%Y%m%d_%H%M%S)}"
EXPERT_DATASET="$ROOT_DIR/data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"

# Test seeds matching expert dataset
SEEDS=(20260902 20260903 20260904)

echo "=================================================="
echo "Expert vs ACT Trajectory Analysis"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Seeds: ${SEEDS[*]}"
echo ""

mkdir -p "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "Analyzing seed $seed..."
    python "$ROOT_DIR/sim/validation/expert_vs_act_trajectory_analysis.py" \
        "$CHECKPOINT" \
        --seed "$seed" \
        --expert-dataset "$EXPERT_DATASET" \
        --max-seconds 10.0 \
        --device cuda \
        --output "$OUTPUT_DIR/trajectory_analysis_seed${seed}.json" \
        2>&1 | tee "$OUTPUT_DIR/trajectory_analysis_seed${seed}.log"
    echo ""
done

echo "=================================================="
echo "Analysis complete. Results in: $OUTPUT_DIR"
echo "=================================================="

# Generate summary
python - <<EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results = []

for seed in ${SEEDS[@]}:
    result_file = output_dir / f"trajectory_analysis_seed{seed}.json"
    if result_file.exists():
        with open(result_file) as f:
            results.append(json.load(f))

if results:
    print("\n" + "="*80)
    print("SUMMARY ACROSS SEEDS")
    print("="*80)

    for stage in ["upsampler", "lowpass", "guard"]:
        deviations = [r["stage_analysis"][stage]["deviation"]["mean"] for r in results]
        print(f"\n{stage.upper()}:")
        print(f"  Mean deviation: {sum(deviations)/len(deviations):.6f} ± {max(deviations)-min(deviations):.6f}")

        if stage == "guard":
            total_clamps = {}
            for r in results:
                clamps = r["stage_analysis"]["guard"]["clamps"]
                for k, v in clamps.items():
                    total_clamps[k] = total_clamps.get(k, 0) + v
            print(f"  Total clamps across seeds:")
            for k, v in total_clamps.items():
                print(f"    {k}: {v}")

EOF
