"""Checkout-relative defaults and explicit relocation of recorded provenance."""
import json
import os
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT.parents[1]
REAL_ROOT = Path(os.environ.get('OPENARM_ACT_REAL_DATA', WORKSPACE / 'data/lerobot/real')).expanduser().resolve()


def recorded_path(value) -> Path:
    """Resolve relative paths, or an explicitly configured old-prefix mapping.

    OPENARM_ACT_PATH_MAP is a JSON object {"/old/root": "/new/root"}. Absolute
    provenance is never silently redirected to a different local dataset.
    """
    path = Path(value).expanduser()
    mapping = json.loads(os.environ.get('OPENARM_ACT_PATH_MAP', '{}'))
    if not isinstance(mapping, dict):
        raise ValueError('OPENARM_ACT_PATH_MAP must be a JSON object')
    for old, new in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
        try:
            suffix = path.relative_to(Path(old).expanduser())
        except ValueError:
            continue
        target = Path(new).expanduser()
        return (target if target.is_absolute() else WORKSPACE / target) / suffix
    return path if path.is_absolute() else WORKSPACE / path
