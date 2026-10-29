"""Read frozen real reference phases locally; no Dora/old experiment imports."""
import json
from pathlib import Path

import numpy as np

from .data import EpisodeDataset, REAL_ROOT, FPS


class SavedDataset:
    """Compatibility for the historical close/open timing report, images unused."""
    def __init__(self, split, images=False):
        if images:
            raise ValueError("Reference phase reader is numeric-only")
        ds = EpisodeDataset(REAL_ROOT / split, domain="real", images=False)
        self.episodes = []
        for e, entry in zip(ds.episodes, ds.manifest["episodes"]):
            source = Path(entry["source_episode_path"])
            records = [json.loads(line) for line in (source / "events.jsonl").read_text().splitlines() if line.strip()]
            events = {r["detail"]["name"]: r["timestamp_s"] - entry["start_timestamp_s"]
                      for r in records if r.get("phase") == "timeline_event"}
            a, phases = e["a"], {}
            for phase, following in (("close", "carry"), ("open", "return")):
                start = max(0, int(np.ceil(events[phase] * FPS)))
                end = min(len(a) - 1, int(np.floor(events[following] * FPS)))
                if start >= end:
                    raise ValueError("Invalid real reference phase interval")
                g0, g1 = float(a[start, 7]), float(a[end, 7])
                if abs(g1 - g0) <= .3 or (g1 > g0) != (phase == "close"):
                    raise ValueError("Invalid real reference gripper transition")
                progress = (a[:, 7] - g0) / (g1 - g0)
                crossings = {}
                for fraction in (.01, .1, .5, .9, .99):
                    flags = progress[start:end + 1] >= fraction
                    hits = np.flatnonzero(flags[:-2] & flags[1:-1] & flags[2:])
                    if not len(hits):
                        raise ValueError("Missing real reference crossing")
                    crossings[str(fraction)] = start + int(hits[0])
                phases[phase] = dict(start=start, end=end, lo=max(0, start - 50),
                    hi=min(len(a), crossings["0.99"] + 16), g0=g0, g1=g1, crossings=crossings)
            self.episodes.append({**e, "source": source, "events": events, "phases": phases})
