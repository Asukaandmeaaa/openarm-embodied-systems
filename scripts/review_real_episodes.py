#!/usr/bin/env python3
"""Interactive, offline reviewer for real-robot ACT episodes.

No ROS or robot I/O is used.  Two synchronized camera views and the measured
versus controller-reference joints are shown using the same renderer as
``replay_real_episode.py``.  With ``--write``, the reviewer records explicit
human decisions in each episode's metadata.json.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2

from replay_real_episode import Episode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "openarm_kinesthetic_data/collections/episodes_pending_review"


def atomic_write(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix="metadata.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def decision(meta: dict, status: str, notes: str) -> dict:
    out = dict(meta)
    out["review_status"] = status
    out["operator_marked_success"] = status == "approved"
    out["terminal_label"] = "success" if status == "approved" else "failure" if status == "rejected" else None
    previous = str(meta.get("review_notes") or "")
    out["review_notes"] = notes or previous
    history = list(meta.get("review_history") or [])
    history.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": status, "notes": notes})
    out["review_history"] = history
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--pattern", default="20261008_*", help="episode directory glob")
    ap.add_argument("--episode", action="append", help="review only this ID (repeatable)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--write", action="store_true", help="write decisions to metadata.json")
    ap.add_argument("--check", action="store_true", help="validate files and list episodes without opening a GUI")
    args = ap.parse_args()
    if args.speed <= 0 or args.fps <= 0:
        ap.error("--speed and --fps must be positive")

    paths = sorted(p.parent for p in args.source.glob(f"{args.pattern}/metadata.json"))
    if args.episode:
        wanted = set(args.episode)
        paths = [p for p in paths if p.name in wanted]
        missing = sorted(wanted - {p.name for p in paths})
        if missing:
            ap.error(f"episode IDs not found: {missing}")
    if not paths:
        ap.error("no episodes found")

    group_members = {}
    for path in paths:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        group = meta.get("source_trajectory_trial_id") or meta.get("trial_id") or path.name
        group_members.setdefault(str(group), []).append(path.name)
    for members in group_members.values():
        members.sort()

    print(f"Episodes: {len(paths)} | write={'ON' if args.write else 'OFF'}")
    if args.check:
        for path in paths:
            episode = Episode(path)
            print(f"PASS {path.name} group={episode.meta.get('source_trajectory_trial_id') or episode.meta.get('trial_id')} review={episode.meta.get('review_status')}")
        return

    print("Review each complete trajectory. Check: cube is visibly grasped; it enters the box; gripper opens only after placement; robot returns HOME; no obvious camera/data corruption.")
    print("Keys: SPACE pause | A/D seek 1s | R restart | Y approve | X reject | P leave pending | N next | Q quit")
    window = "OpenArm real-episode review"
    try:
        for number, path in enumerate(paths, 1):
            episode = Episode(path)
            group = str(episode.meta.get("source_trajectory_trial_id") or episode.meta.get("trial_id") or path.name)
            siblings = ",".join(group_members[group])
            print(f"\n[{number}/{len(paths)}] {path.name} source_group={group} replays={siblings}")
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 800)
            seconds, paused, previous = 0.0, False, cv2.getTickCount() / cv2.getTickFrequency()
            control = {"seek": None, "updating": False}

            def seek(value):
                if not control["updating"]:
                    control["seek"] = value / 1000.0 * episode.duration

            cv2.createTrackbar("seek", window, 0, 1000, seek)
            move_next = False
            while True:
                now = cv2.getTickCount() / cv2.getTickFrequency()
                if control["seek"] is not None:
                    seconds, control["seek"] = control["seek"], None
                elif not paused:
                    seconds = min(episode.duration, seconds + (now - previous) * args.speed)
                previous = now
                if seconds >= episode.duration:
                    paused = True
                frame = cv2.copyMakeBorder(episode.render(seconds), 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
                cv2.putText(frame, f"[{number}/{len(paths)}] source group: {group} | replays: {siblings}", (15, 735), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 220, 80), 1, cv2.LINE_AA)
                cv2.putText(frame, "Y=usable/approve  X=not usable/reject  P=pending  N=next  Q=quit", (15, 758), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)
                control["updating"] = True
                cv2.setTrackbarPos("seek", window, round(seconds / episode.duration * 1000))
                control["updating"] = False
                key = cv2.waitKey(max(1, round(1000 / args.fps))) & 0xFF
                if key in (27, ord("q")):
                    return
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("a"), ord("d")):
                    seconds = max(0.0, min(episode.duration, seconds + (-1 if key == ord("a") else 1)))
                    paused = True
                elif key == ord("r"):
                    seconds, paused = 0.0, False
                elif key in (ord("y"), ord("x"), ord("p"), ord("n")):
                    if key == ord("n"):
                        move_next = True
                        break
                    status = {ord("y"): "approved", ord("x"): "rejected", ord("p"): "pending"}[key]
                    notes = input(f"Notes for {path.name} [{status}] (optional): ").strip()
                    if args.write:
                        meta_path = path / "metadata.json"
                        current = json.loads(meta_path.read_text(encoding="utf-8"))
                        atomic_write(meta_path, decision(current, status, notes))
                        print(f"SAVED {path.name}: {status}")
                    else:
                        print(f"DRY RUN {path.name}: {status} (use --write to save)")
                    move_next = True
                    break
            cv2.destroyWindow(window)
            if move_next:
                continue
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
