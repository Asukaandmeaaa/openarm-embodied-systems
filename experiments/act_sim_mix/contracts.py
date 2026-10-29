"""Hardware-free data constants, shared without importing optional PyTorch."""
import hashlib
from pathlib import Path

FPS = 30
HORIZON = 100
CAMS = ('observation.images.ceiling', 'observation.images.wrist_right')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def robot_asset_binding() -> dict[str, str]:
    from .paths import WORKSPACE
    root = WORKSPACE / 'official/openarm_mujoco/v2'
    return {str(p.relative_to(WORKSPACE)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
