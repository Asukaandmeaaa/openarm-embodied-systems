"""Offline recording contract; deliberately has no CAN transmit or motor imports."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAMES = [f'openarm_{s}_joint{i}' if i < 8 else f'openarm_{s}_finger_joint1'
         for s in ('right', 'left') for i in range(1, 9)]
MOTORS = ['DM8009', 'DM8009', 'DM4340', 'DM4340'] + ['DM4310'] * 4
LIMITS = {'DM8009': (12.5, 45., 54.), 'DM4340': (12.5, 10., 28.),
          'DM4310': (12.5, 30., 10.)}


def preserve_nonfinite(value):
    """Losslessly tag ROS metadata inf/nan; never reinterpret it as a position."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'__ros_nonfinite_float__': repr(value)}
    if isinstance(value, dict):
        return {k: preserve_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [preserve_nonfinite(v) for v in value]
    return value


def check_driver_contract(config):
    motor = config['motor_config']
    if (motor['types'] != MOTORS or motor['send_ids'] != list(range(1,9))
            or motor['recv_ids'] != list(range(0x11,0x19))
            or config.get('gripper_posforce') is not False
            or any(config['joint_offsets'][s] != [0.]*8 for s in ('right_arm','left_arm'))):
        raise ValueError('capture decoder requires v2 MIT axes, ascending IDs, and zero ROS offsets')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def decode(can_id, payload):
    """Match vendored openarm_can MIT encoding; default v2 ascending IDs only.

    Positions are raw motor radians, also ROS coordinates for the v2 pinch
    gripper. Other motor models, IDs and gripper transmissions are unsupported.
    """
    if can_id & 0xE0000000 or len(payload) != 8:
        return None
    b = payload
    if 1 <= can_id <= 8:
        axis, kind = can_id-1, 'command'
        if b[:7] == bytes([255])*7 and b[7] in (0xFC, 0xFD, 0xFE):
            return {'axis': axis, 'kind': 'lifecycle',
                    'operation': {0xFC:'enable', 0xFD:'disable', 0xFE:'set_zero'}[b[7]]}
    elif 0x11 <= can_id <= 0x18:
        axis, kind = can_id-0x11, 'feedback'
        # Parameter replies share the response identifier. Do not decode as state.
        if b[2] in (0x33, 0x55) and b[0] == axis+1 and b[1] == 0:
            return None
    else:
        return None
    p, v, tau = LIMITS[MOTORS[axis]]
    def sym(value, limit, bits):
        return value * (2*limit)/((1 << bits)-1) - limit
    out = {'axis': axis, 'kind': kind, 'motor_type': MOTORS[axis]}
    if kind == 'command':
        out.update(q=sym(b[0]*256+b[1], p, 16),
                   dq=sym(b[2]*16+(b[3] >> 4), v, 12),
                   kp=((b[3] & 15)*256+b[4])*500/4095,
                   kd=(b[5]*16+(b[6] >> 4))*5/4095,
                   torque=sym((b[6] & 15)*256+b[7], tau, 12))
    else:
        out.update(q=sym(b[1]*256+b[2], p, 16),
                   dq=sym(b[3]*16+(b[4] >> 4), v, 12),
                   torque=sym((b[4] & 15)*256+b[5], tau, 12),
                   status=b[0] >> 4, temperature_mos=b[6], temperature_rotor=b[7])
    return out


def validate_bundle(bundle):
    if bundle.get('schema') != 'openarm_recorded_home_v1' or bundle.get('joint_names') != NAMES:
        raise ValueError('unsupported schema or joint ordering')
    t = np.asarray(bundle['time_s'], dtype=float)
    q = np.asarray(bundle['positions_rad'], dtype=float)
    if (t.ndim != 1 or len(t) < 2 or q.shape != (len(t), 16)
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(t))
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError('trajectory must have finite 16D positions and strictly increasing times starting at zero')
    if bundle.get('interpolation') != 'piecewise_linear_recorded_samples':
        raise ValueError('unsupported interpolation')
    for key in ('measured_start_rad', 'measured_end_rad'):
        value = np.asarray(bundle[key], dtype=float)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f'invalid {key}')
    return t, q


def check_start(bundle, measured, tolerance=.03, active_indices=None):
    """No automatic alignment trajectory: mismatch is a refusal, not a motion."""
    _, q = validate_bundle(bundle)
    measured = np.asarray(measured, dtype=float)
    if measured.shape != (16,) or not np.all(np.isfinite(measured)):
        raise ValueError('fresh feedback must contain all 16 finite joints')
    error = np.maximum(np.abs(measured-q[0]),
                       np.abs(measured-np.array(bundle['measured_start_rad'])))
    if active_indices is not None:
        error = error[np.asarray(active_indices)]
    if float(error.max()) > tolerance:
        axis = int(error.argmax()) if active_indices is None else int(active_indices[error.argmax()])
        raise ValueError(f'start mismatch at {NAMES[axis]}: {error.max():.6f} rad')
    return float(error.max())


def sample(bundle, elapsed):
    # Each neighboring recorded point is retained. No zero/side/HOME shortcuts.
    t, q = validate_bundle(bundle)
    return np.array([np.interp(elapsed, t, q[:, j]) for j in range(16)])


def dense_samples(bundle, max_dt=.005, max_dq=.002):
    """Subdivide EVERY recorded segment, including short transients."""
    t, q = validate_bundle(bundle)
    yield 0., q[0]
    for i in range(1, len(t)):
        count = max(1, int(np.ceil((t[i]-t[i-1])/max_dt)),
                    int(np.ceil(np.max(np.abs(q[i]-q[i-1]))/max_dq)))
        if count > 100000:
            raise ValueError('unreasonable segment subdivision')
        for k in range(1, count+1):
            a = k/count
            yield float(t[i-1]+a*(t[i]-t[i-1])), q[i-1]+a*(q[i]-q[i-1])
