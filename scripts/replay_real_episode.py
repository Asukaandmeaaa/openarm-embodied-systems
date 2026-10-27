#!/usr/bin/env python3
"""Offline timestamp-aligned review of real OpenArm episodes; no ROS or hardware I/O."""
import argparse
from bisect import bisect_right
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'openarm_kinesthetic_data/collections/episodes_pending_review'


def read_rows(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not math.isfinite(r['timestamp_s']) for r in rows):
        raise ValueError(f'Nonfinite timestamp: {path}')
    return sorted(rows, key=lambda r: r['timestamp_s'])


class Stream:
    def __init__(self, rows):
        self.rows = rows
        self.times = [r['timestamp_s'] for r in rows]

    def at(self, timestamp):
        index = bisect_right(self.times, timestamp) - 1
        return self.rows[index] if index >= 0 else None


def put(canvas, text, x, y, color=(220, 220, 220), scale=0.52):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Episode:
    def __init__(self, path):
        self.path = path.resolve()
        self.meta = json.loads((path / 'metadata.json').read_text())
        rows = read_rows(path / 'camera.jsonl')
        self.cameras = {key: Stream([r for r in rows if r['key'] == key])
                        for key in ('ceiling', 'wrist_right')}
        if any(not s.rows for s in self.cameras.values()):
            raise ValueError('Both ceiling and wrist_right camera streams are required')
        for row in rows:
            target = (self.path / row['path']).resolve()
            if not target.is_relative_to(self.path) or not target.is_file():
                raise ValueError(f'Missing or invalid image: {target}')
        self.state = Stream(read_rows(path / 'joint_observation.jsonl'))
        self.action = Stream(read_rows(path / 'demonstration_action.jsonl'))
        self.events = Stream(read_rows(path / 'gripper_event.jsonl'))
        streams = [*self.cameras.values(), self.state, self.action, self.events]
        self.start = min(s.times[0] for s in streams if s.times)
        self.end = max(s.times[-1] for s in streams if s.times)
        self.duration = self.end - self.start
        self.cache = {}
        print(f'{path.name}: {self.duration:.3f}s, review={self.meta.get("review_status")}')
        for key, stream in zip(('ceiling', 'wrist_right', 'state', 'action', 'events'), streams):
            print(f'  {key}: {len(stream.rows)} records')

    def render(self, seconds):
        timestamp = self.start + seconds
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for index, (key, stream) in enumerate(self.cameras.items()):
            row = stream.at(timestamp)
            x = index * 640
            if row is None:
                put(canvas, f'{key}: no sample yet', x + 12, 40)
                continue
            image_path = self.path / row['path']
            if self.cache.get(key, (None,))[0] != image_path:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f'Cannot decode {image_path}')
                h, w = img.shape[:2]
                scale = min(640/w, 360/h)
                resized = cv2.resize(img, (round(w*scale), round(h*scale)))
                tile = np.zeros((360, 640, 3), dtype=np.uint8)
                h, w = resized.shape[:2]
                tile[(360-h)//2:(360-h)//2+h, (640-w)//2:(640-w)//2+w] = resized
                self.cache[key] = (image_path, tile)
            canvas[:360, x:x+640] = self.cache[key][1]
            age = timestamp - row['timestamp_s']
            cv2.rectangle(canvas, (x, 0), (x+640, 30), (0, 0, 0), -1)
            put(canvas, f'{key}  sample age={age*1000:.0f}ms' + ('  STALE' if age > .15 else ''),
                x+10, 21, (0, 100, 255) if age > .15 else (0, 255, 0))
        put(canvas, f'{self.path.name}   t={seconds:.2f}/{self.duration:.2f}s   review={self.meta.get("review_status")}', 15, 390)
        put(canvas, 'Latest sample at/before playback time; age >150ms is marked STALE. Units: rad.', 15, 415)
        state, action = self.state.at(timestamp), self.action.at(timestamp)
        for index, side in enumerate(('right', 'left')):
            x = 15 + index*640
            put(canvas, f'{side.upper()}       measured       reference       measured-reference', x, 445)
            for j in range(8):
                name = f'openarm_{side}_joint{j+1}' if j < 7 else f'openarm_{side}_finger_joint1'
                def value(row):
                    if row is None or name not in row['order']:
                        return None
                    return row['positions'][row['order'].index(name)]
                q, a = value(state), value(action)
                def fmt(v):
                    return '    --' if v is None else f'{v: .4f}'
                put(canvas, f'{"grip" if j == 7 else "J"+str(j+1):6s}     {fmt(q):>8}        {fmt(a):>8}        {fmt(q-a if q is not None and a is not None else None):>8}', x, 470+j*23)
        def age_text(row):
            if row is None:
                return 'not started'
            age = timestamp-row['timestamp_s']
            return f'{age*1000:.0f}ms' + (' STALE' if age > .15 else '')
        event = self.events.at(timestamp)
        put(canvas, f'State age: {age_text(state)} | Reference age: {age_text(action)} | Last gripper command: {event["event"] if event else "none"}', 15, 665)
        put(canvas, 'SPACE pause/resume | A/D seek -/+1s | R restart | Q/ESC quit | seek bar below', 15, 698)
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', nargs='?', default='20260929_115522_015', help='Episode directory or pending episode ID')
    parser.add_argument('--speed', type=float, default=1)
    parser.add_argument('--fps', type=float, default=30, help='Display/export sampling rate')
    parser.add_argument('--export', type=Path, help='Export a silent MP4 instead of opening a window')
    parser.add_argument('--check', action='store_true', help='Check file references and decode representative frames without GUI')
    args = parser.parse_args()
    if not (math.isfinite(args.speed) and math.isfinite(args.fps)) or args.speed <= 0 or args.fps <= 0:
        parser.error('speed and fps must be positive and finite')
    path = Path(args.episode).expanduser()
    if not path.is_dir():
        path = DEFAULT / args.episode
    episode = Episode(path)
    if args.check:
        for t in np.linspace(0, episode.duration, 12):
            episode.render(float(t))
        print('PASS: all image paths exist; 12 representative composite frames decoded. Not a task-success assessment.')
        return
    if args.export:
        if args.export.exists():
            parser.error(f'Output already exists: {args.export}')
        args.export.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.export), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('Cannot open video writer')
        try:
            for frame in range(math.ceil(episode.duration*args.fps/args.speed)+1):
                writer.write(episode.render(min(frame*args.speed/args.fps, episode.duration)))
        finally:
            writer.release()
        print(f'Exported: {args.export}')
        return
    window = 'OpenArm offline episode review'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 760)
    control = {'seek': None, 'updating': False}
    def seek(value):
        if not control['updating']:
            control['seek'] = value/1000*episode.duration
    cv2.createTrackbar('seek', window, 0, 1000, seek)
    seconds, paused, previous = 0., False, time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if control['seek'] is not None:
                seconds = control['seek']; control['seek'] = None
            elif not paused:
                seconds = min(episode.duration, seconds+(now-previous)*args.speed)
            previous = now
            if seconds >= episode.duration:
                paused = True
            cv2.imshow(window, episode.render(seconds))
            control['updating'] = True
            cv2.setTrackbarPos('seek', window, round(seconds/episode.duration*1000))
            control['updating'] = False
            key = cv2.waitKey(max(1, round(1000/args.fps))) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                paused = not paused
            elif key in (ord('a'), ord('d')):
                seconds = float(np.clip(seconds+(-1 if key == ord('a') else 1), 0, episode.duration))
            elif key == ord('r'):
                seconds, paused = 0., False
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
