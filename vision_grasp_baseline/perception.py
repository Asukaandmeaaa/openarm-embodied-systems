#!/usr/bin/env python3
"""Inspect the current blue-cube/black-bin fixture from aligned RGB-D.

All world coordinates retain the calibration's provenance. This measures the
present fixture; it does not reuse the old elevated-white-box detector.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

# run.sh also invokes this file directly.
if __package__ in (None,''):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vision_grasp_baseline.scene_estimation import stabilize_scene,UnstableSceneError


def camera_transform(calibration, metadata):
    cal = yaml.safe_load(Path(calibration).read_text())
    if (cal['parameters']['robot_base_frame'] != 'world' or
            cal['parameters']['tracking_base_frame'] != 'global_camera_link'):
        raise ValueError('Expected world -> global_camera_link calibration')
    t, q = cal['transform']['translation'], cal['transform']['rotation']
    world_link = np.eye(4)
    world_link[:3, :3] = Rotation.from_quat([q[k] for k in 'xyzw']).as_matrix()
    world_link[:3, 3] = [t[k] for k in 'xyz']
    # Official realsense-ros: camera_link origin is left IR/depth, ROS axes.
    link_depth = np.eye(4)
    link_depth[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    ex = metadata['depth_to_color_extrinsics']
    color_depth = np.eye(4)
    color_depth[:3, :3] = np.asarray(ex['rotation_column_major']).reshape(3, 3, order='F')
    color_depth[:3, 3] = ex['translation_m']
    return world_link @ link_depth @ np.linalg.inv(color_depth)


def fit_plane(points, threshold=.006):
    points = np.asarray(points, dtype=float)
    if len(points) < 300:
        raise ValueError('Too few support-plane points')
    rng = np.random.default_rng(21)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(100):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        mask = abs((points-a) @ n) < threshold
        if mask.sum() > best.sum():
            best = mask
    if best.mean() < .65:
        raise ValueError('No dominant support plane')
    center = np.mean(points[best], axis=0)
    _, _, vt = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    d = -normal @ center
    return normal, d, abs(points[best] @ normal+d)


def rectangle(points):
    xy = np.asarray(points)[:, :2]
    center = np.median(xy, axis=0)
    rect = cv2.minAreaRect((xy-center).astype(np.float32))
    corners = cv2.boxPoints(rect).astype(float)+center
    edges = np.roll(corners, -1, axis=0)-corners
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    yaw = np.arctan2(edge[1], edge[0])
    # One canonical representation for square-grasp symmetry, no 90deg jumps.
    yaw = (yaw+np.pi/4) % (np.pi/2)-np.pi/4
    r = Rotation.from_euler('z', yaw).as_matrix()[:2, :2]
    local = (xy-center) @ r
    low, high = np.percentile(local, [2, 98], axis=0)
    c = center+((low+high)/2) @ r.T
    return c, high-low, float(yaw)


def contour(mask, minimum, allow_border=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = mask.shape
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cv2.contourArea(c) < minimum:
            continue
        if not allow_border and (x < 3 or y < 3 or x+cw > w-3 or y+ch > h-3):
            continue
        candidates.append(c)
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one component, got {len(candidates)}')
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, candidates, -1, 255, -1)
    return filled > 0, candidates[0]


def cube_pixels(hsv):
    blue = cv2.inRange(hsv, (90, 150, 65), (130, 255, 255))
    associated = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    mask, outline = contour(associated, 1000)
    return mask & (blue > 0), outline


def analyze(directory, calibration,*,stationary=False):
    p = Path(directory)
    m = json.loads((p/'capture.json').read_text())
    w_c = camera_transform(calibration, m)
    rows, annotated = [], None
    for frame in m['frames']:
        index = frame['index']
        im = cv2.imread(str(p/f'color_{index:03d}.png'))
        z = np.load(p/f'depth_{index:03d}.npy') * m['depth_scale_m']
        i = m['color_intrinsics']
        v, u = np.indices(z.shape)
        optical = np.stack(((u-i['ppx'])*z/i['fx'], (v-i['ppy'])*z/i['fy'], z), axis=-1)
        world = optical @ w_c[:3, :3].T+w_c[:3, 3]
        valid = (z > .15) & (z < 1.5)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        white = valid & (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 130)
        normal, d, residual = fit_plane(world[white][::8])
        height = world @ normal+d
        # The wrist cable can split this one physical cube's color silhouette.
        # Bridge a narrow image gap for association, then use only genuinely
        # blue pixels for 3-D fitting so cable depth never becomes cube height.
        cube_mask, cube_contour = cube_pixels(hsv)
        cube_points = world[cube_mask & valid]
        cube_heights = height[cube_mask & valid]
        top = float(np.percentile(cube_heights, 85))
        if not .02 < top < .08:
            raise ValueError(f'Unexpected measured cube height: {top}')
        upper = cube_points[abs(cube_heights-top) < .003]
        cube_xy, cube_size, cube_yaw = rectangle(upper)
        cube_support_z = float(-(normal[:2] @ cube_xy+d)/normal[2])
        dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 75))
        bin_mask, bin_contour = contour(dark, 3000)
        # Contents of the open bin are not part of its rim geometry.
        bin_mask &= dark > 0
        bin_heights = height[bin_mask & valid]
        rim = float(np.percentile(bin_heights, 95))
        if not .02 < rim < .10:
            raise ValueError(f'Unexpected measured bin rim height: {rim}')
        bin_upper = world[bin_mask & valid & (height > rim-.008) & (height < rim+.008)]
        bin_xy, bin_size, bin_yaw = rectangle(bin_upper)
        bin_support_z = float(-(normal[:2] @ bin_xy+d)/normal[2])
        # A small clearance above the rim avoids fingers entering the bin.
        # This is a release/drop target, not a measured bottom contact pose.
        release_z = bin_support_z+rim+.015+top/2
        table_points = world[white & (abs(height) < .006)]
        table_xy, table_size, table_yaw = rectangle(table_points[::4])
        table_z = float(-(normal[:2] @ table_xy+d)/normal[2])
        row = dict(index=index, table=dict(center_xy=table_xy.tolist(), size_xy=table_size.tolist(),
                   top_z=table_z, yaw=table_yaw, normal=normal.tolist(), plane_d=float(d),
                   plane_residual_p95_m=float(np.percentile(residual,95)),
                   tilt_deg=float(np.degrees(np.arccos(normal[2])))),
                   cube=dict(center=[*cube_xy.tolist(),cube_support_z+top/2],
                             size=[*cube_size.tolist(),top], yaw=cube_yaw, support_z=cube_support_z),
                   bin=dict(center_xy=bin_xy.tolist(), outer_size_xy=bin_size.tolist(),
                            yaw=bin_yaw, support_z=bin_support_z, rim_height=rim,
                            release_object_center_z=release_z))
        rows.append(row)
        annotated = im.copy()
        cv2.drawContours(annotated, [cube_contour], -1, (0,255,0), 2)
        cv2.drawContours(annotated, [bin_contour], -1, (0,180,255), 2)
        cv2.putText(annotated, f'cube height {top*1000:.1f}mm; rim {rim*1000:.1f}mm',
                    (12,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)
        cv2.putText(annotated, 'WORLD TF: pending hover verification',
                    (12,48), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)
    latest = rows[-1]
    stability = dict(cube_center_std_m=np.std([r['cube']['center'] for r in rows],axis=0).tolist(),
                     bin_center_std_m=np.std([r['bin']['center_xy'] for r in rows],axis=0).tolist())
    result = dict(schema_version=1, capture=str(p.resolve()), calibration=str(Path(calibration).resolve()),
                  calibration_status='operator_accepts_provisional_for_hover_check',
                  world_from_color_optical=w_c.tolist(), serial=m['serial'],
                  capture_wall_ns=m['frames'][-1]['wall_ns'], measured=deepcopy(latest), stability=stability,
                  frames=rows, grasp_execution_ready=False,
                  limitations=['Base may have moved; verify world XY before descent.',
                               'Partial tabletop extent is measured; table legs are not observed.',
                               'Bin inner aperture/floor thickness are not yet measured.',
                               'Color thresholds cover this blue cube and dark bin only.'])
    if len(rows)>=5:
        try:result=stabilize_scene(result,stationary=stationary)
        except UnstableSceneError as exc:
            result['detection_error']=str(exc)
            (p/'scene_unstable.json').write_text(json.dumps(result,indent=2)+'\n')
            cv2.imwrite(str(p/'detection_unstable.png'),annotated)
            raise
    (p/'scene.json').write_text(json.dumps(result,indent=2)+'\n')
    cv2.imwrite(str(p/'detection.png'), annotated)
    print(json.dumps(dict(scene=str(p/'scene.json'),measured=result['measured'],stability=stability),indent=2))
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',required=True)
    ap.add_argument('--calibration',required=True)
    ap.add_argument('--stationary-fixture',action='store_true')
    a=ap.parse_args()
    try:analyze(a.capture,a.calibration,stationary=a.stationary_fixture)
    except UnstableSceneError as exc:
        print(str(exc),flush=True)
        raise SystemExit(3)
