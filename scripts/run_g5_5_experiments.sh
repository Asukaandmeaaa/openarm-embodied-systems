#!/usr/bin/env bash
# Run G5.5 Executor Semantic Equivalence Gate experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if conda vla environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: No conda environment detected."
    echo "Please activate the vla environment first:"
    echo "  conda activate vla"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
elif [ "$CONDA_DEFAULT_ENV" != "vla" ]; then
    echo "WARNING: Current conda environment is '$CONDA_DEFAULT_ENV', not 'vla'."
    echo "Please activate the vla environment first:"
    echo "  conda activate vla"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default parameters
SEED=20260902
DATASET="$ROOT_DIR/data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
CHECKPOINT="$ROOT_DIR/outputs/act_openarm_cube_pick_smoke_g5_500/checkpoints/last"
SCENE="$ROOT_DIR/official/openarm_mujoco/v2/cell/demo.xml"
PARAMS="$ROOT_DIR/configs/collection/openarm_cube_pick_expert_params.yaml"
OUTPUT_DIR="$ROOT_DIR/results/g5_5_executor_equivalence"
DEVICE="cuda"
MODE="full"  # "smoke" or "full"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            MODE="smoke"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke          Run 10s smoke test"
            echo "  --full           Run full 45s episode (default)"
            echo "  --seed SEED      Seed for deterministic reset (default: 20260902)"
            echo "  --device DEVICE  Device for ACT policy (default: cuda)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "G5.5 Executor Semantic Equivalence Gate"
echo "=================================================="
echo "Mode: $MODE"
echo "Seed: $SEED"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check dependencies
if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

if [ ! -e "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$SCENE" ]; then
    echo "ERROR: Scene not found: $SCENE"
    exit 1
fi

# Run tests first
echo "Running unit tests..."
python "$ROOT_DIR/tests/test_g5_5_executor_equivalence.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failed. Fix issues before running experiments."
    exit 1
fi
echo ""

# Run experiments
SMOKE_FLAG=""
if [ "$MODE" == "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi

echo "Running G5.5 experiments..."
python "$ROOT_DIR/sim/validation/g5_5_executor_equivalence.py" \
    --experiment all \
    --seed "$SEED" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --scene "$SCENE" \
    --params "$PARAMS" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    $SMOKE_FLAG

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "G5.5 experiments completed successfully"
    echo "=================================================="
    echo "Results saved to: $OUTPUT_DIR/seed_${SEED}_${MODE}/"
    echo ""
    echo "Next steps:"
    echo "  1. Review summary.json for gate status"
    echo "  2. Check individual experiment results"
    echo "  3. Generate visualizations if needed"
else
    echo ""
    echo "=================================================="
    echo "G5.5 experiments failed with exit code $EXIT_CODE"
    echo "=================================================="
    echo "Check logs above for error details"
fi

exit $EXIT_CODE
