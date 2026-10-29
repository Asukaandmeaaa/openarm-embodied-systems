"""Datasets and domain sampling for the 100-step mixed ACT experiment.

Real frames are read from the frozen physical dataset.  Simulation frames are
read from the compact converted splits.  Both use episode-local 100-step
labels with final-frame padding; labels never cross an episode boundary.
Images remain compressed in memory and are decoded per sampled frame.
"""
from __future__ import annotations

from collections import OrderedDict
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from .paths import REAL_ROOT
HORIZON = 100
FPS = 30
CAMS = ("observation.images.ceiling", "observation.images.wrist_right")


def digest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class EpisodeDataset(torch.utils.data.Dataset):
    """Episode-local 100-step windows with compressed-image LRU caching."""

    def __init__(self, root: Path, *, domain: str, images: bool = True, cache_episodes: int = 8):
        self.root = Path(root)
        self.domain = domain
        self.images = images
        self.cache_episodes = cache_episodes
        self.cache: OrderedDict[int, dict] = OrderedDict()
        manifest_path = self.root / "meta/visual_expert_manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.episodes = []
        offset = 0
        for number, entry in enumerate(self.manifest["episodes"]):
            path = self.root / f"data/chunk-000/file-{number:03d}.parquet"
            table = pq.read_table(path, columns=["action", "observation.state"])
            a = np.array(table["action"].to_pylist(), dtype=np.float32)
            q = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
            frame_count = int(entry.get("frame_count", entry.get("converted_frames", len(a))))
            if a.shape != (frame_count, 16) or q.shape != (frame_count, 16):
                raise ValueError(f"bad numeric shape in {path}")
            if not np.isfinite(a).all() or not np.isfinite(q).all():
                raise ValueError(f"non-finite data in {path}")
            e = dict(
                number=number,
                path=path,
                length=len(a),
                offset=offset,
                a=a,
                q=q,
                group=entry.get("group_id", str(number)),
                recovery_type=entry.get("recovery_type"),
                source=entry.get("path", str(path)),
                domain=entry.get("domain", domain),
            )
            self.episodes.append(e)
            offset += len(a)
        self.ends = np.cumsum([e["length"] for e in self.episodes]) if self.episodes else np.array([])
        self.total = int(self.ends[-1]) if len(self.ends) else 0

    def __len__(self) -> int:
        return self.total

    def locate(self, index: int):
        episode = int(np.searchsorted(self.ends, index, side="right"))
        e = self.episodes[episode]
        return e, int(index - e["offset"])

    def _table(self, local_episode: int) -> dict:
        if local_episode in self.cache:
            self.cache.move_to_end(local_episode)
            return self.cache[local_episode]
        e = self.episodes[local_episode]
        table = pq.read_table(e["path"], columns=list(CAMS))
        cached = {
            role: [row["bytes"] for row in table[role].to_pylist()] for role in CAMS
        }
        self.cache[local_episode] = cached
        self.cache.move_to_end(local_episode)
        while len(self.cache) > self.cache_episodes:
            self.cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int):
        index = int(index)
        local_episode = int(np.searchsorted(self.ends, index, side="right"))
        e = self.episodes[local_episode]
        frame = int(index - e["offset"])
        future = frame + np.arange(HORIZON)
        labels = np.minimum(future, e["length"] - 1)
        pad = future >= e["length"]
        batch = {
            "observation.state": torch.from_numpy(e["q"][frame].copy()),
            "action": torch.from_numpy(e["a"][labels].copy()),
            "action_is_pad": torch.from_numpy(pad.astype(bool)),
            "index": torch.tensor(int(index)),
            "episode_index": torch.tensor(e["number"]),
            "frame_index": torch.tensor(frame),
        }
        if self.images:
            cached = self._table(local_episode)
            for role in CAMS:
                image = np.asarray(
                    Image.open(io.BytesIO(cached[role][frame])).convert("RGB")
                )
                if image.shape != (480, 640, 3):
                    raise ValueError(f"bad image shape in {e['path']}")
                batch[role] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        return batch

    def episode_indices(self) -> list[int]:
        return list(range(len(self.episodes)))


class MixedSampler:
    """Fixed per-batch domain composition: real 50%, normal sim 25%, recovery 25%."""

    def __init__(self, real: EpisodeDataset, normal: EpisodeDataset, recovery: EpisodeDataset, *, batch_size: int = 8):
        if batch_size % 4 != 0:
            raise ValueError("batch_size must be divisible by 4 for 50/25/25 sampling")
        self.datasets = {"real": real, "sim_normal": normal, "sim_recovery": recovery}
        self.batch_size = batch_size
        self.counts = {k: np.zeros(len(v.episodes), dtype=np.int64) for k, v in self.datasets.items()}
        self.domain_draws = {k: 0 for k in self.datasets}
        self.recipe = {
            "real": 0.5,
            "sim_normal": 0.25,
            "sim_recovery": 0.25,
            "within_domain": "uniform frame index with replacement",
            "batch_size": batch_size,
        }

    def sample(self, rng: np.random.Generator) -> list[int]:
        sizes = {
            "real": self.batch_size * 2 // 4,
            "sim_normal": self.batch_size // 4,
            "sim_recovery": self.batch_size // 4,
        }
        indices: list[int] = []
        for name, size in sizes.items():
            ds = self.datasets[name]
            if len(ds) == 0:
                raise ValueError(f"empty domain {name}")
            local = rng.choice(len(ds), size=size, replace=True)
            for li in local:
                li = int(li)
                episode, _ = ds.locate(li)
                self.counts[name][episode["number"]] += 1
                indices.append(li)
            self.domain_draws[name] += size
        order = rng.permutation(len(indices))
        return [indices[i] for i in order]

    def batch(self, rng: np.random.Generator):
        """Return a collated batch and the source domain for each sample."""
        sizes = {
            "real": self.batch_size * 2 // 4,
            "sim_normal": self.batch_size // 4,
            "sim_recovery": self.batch_size // 4,
        }
        owned = []
        for name, size in sizes.items():
            ds = self.datasets[name]
            local = rng.choice(len(ds), size=size, replace=True)
            for li in local:
                li = int(li)
                local_episode = int(np.searchsorted(ds.ends, li, side="right"))
                self.counts[name][local_episode] += 1
                self.domain_draws[name] += 1
                owned.append((name, ds[li]))
        order = rng.permutation(len(owned))
        items = [owned[i] for i in order]
        batch = {
            key: torch.stack([item[1][key] for item in items])
            for key in items[0][1]
            if key in ("observation.state", "action", "action_is_pad", *CAMS)
        }
        batch["index"] = torch.tensor([int(item[1]["index"]) for item in items])
        batch["episode_index"] = torch.tensor([int(item[1]["episode_index"]) for item in items])
        batch["frame_index"] = torch.tensor([int(item[1]["frame_index"]) for item in items])
        batch["_domains"] = [item[0] for item in items]
        return batch


def load_datasets(converted: Path) -> tuple[EpisodeDataset, EpisodeDataset, EpisodeDataset]:
    real = EpisodeDataset(REAL_ROOT / "train", domain="real")
    normal = EpisodeDataset(
        Path(converted) / "train", domain="sim_normal"
    )
    recovery = EpisodeDataset(
        Path(converted) / "train", domain="sim_recovery"
    )
    # Split the converted train root by the manifest's source domain.
    normal.episodes = [e for e in normal.episodes if e["domain"] == "normal"]
    recovery.episodes = [e for e in recovery.episodes if e["domain"] == "recovery"]
    # Rebuild offsets for the two views over the same parquet files.
    for dataset in (normal, recovery):
        offset = 0
        for e in dataset.episodes:
            e["offset"] = offset
            offset += e["length"]
        dataset.ends = np.cumsum([e["length"] for e in dataset.episodes])
        dataset.total = int(dataset.ends[-1]) if len(dataset.ends) else 0
    return real, normal, recovery


def collate(samples: list[dict]) -> dict:
    from torch.utils.data import default_collate

    return default_collate(samples)
