"""Time-aligned ACT ensemble and analytic q/qd. Pure code, no device access.

Each prediction lives on its ORIGINAL observation clock. Smooth admission and
expiry weights make adding/removing a chunk C1 continuous. No recurrent blend
of blends: every contributor retains its own timestamp and expires at 0.5 s.
"""
import numpy as np

SCHEDULE = 'rolling_time_aligned_v1'
INTERVAL_NS = 33_333_333
PREDICTED_STEPS = 50
QUERY_PERIOD_NS = 100_000_000
MAX_AGE_NS = 500_000_000
EXPIRY_NS = MAX_AGE_NS - 20_000_000  # leave time for guard/logging/dispatch
FADE_S = .10
START_BLEND_S = .20


def contract():
    return dict(name=SCHEDULE, predicted_steps=PREDICTED_STEPS, executed_steps=None,
                interval_ns=INTERVAL_NS, query_period_ns=QUERY_PERIOD_NS,
                next_inference='fresh observations while previous predictions execute',
                time_origin='original observation_monotonic_ns',
                blending='oldest fresh tail priority, smooth time-aligned handover to newer chunks',
                admission_and_expiry_fade_s=FADE_S, start_blend_s=START_BLEND_S,
                maximum_contributor_age_ns=MAX_AGE_NS,
                contribution_expiry_ns=EXPIRY_NS,
                velocity='analytic derivative of the same blended position curve',
                lowpass='no separate biquad; smooth C1 ensemble replaces it')


def full_chunk(actions):
    q = np.asarray(actions['positions'], dtype=float)
    if q.shape != (PREDICTED_STEPS, 16) or not np.isfinite(q).all():
        raise ValueError('rolling ACT requires a finite 50x16 prediction')
    if int(actions['interval']) != INTERVAL_NS:
        raise ValueError('rolling ACT requires original 30 Hz timing')
    return q


def smoothstep(seconds, duration):
    u = float(np.clip(seconds / duration, 0., 1.))
    return u*u*(3-2*u), 6*u*(1-u)/duration if 0 < u < 1 else 0.


class Chunk:
    def __init__(self, positions, metadata, received_ns):
        self.q = full_chunk(dict(positions=positions, interval=metadata['interval']))[:, :8].copy()
        if metadata.get('action_schedule') != SCHEDULE or not metadata.get('queue_chunk_id'):
            raise ValueError('unexpected rolling action schedule/id')
        self.metadata = dict(metadata)
        self.origin_ns = int(metadata['observation_monotonic_ns'])
        self.received_ns = int(received_ns)
        self.dt = INTERVAL_NS/1e9
        # Same monotonicity-aware Hermite slopes as the OpenArm executor.
        secants = np.diff(self.q, axis=0)/self.dt
        self.slopes = np.empty_like(self.q)
        self.slopes[0], self.slopes[-1] = secants[0], secants[-1]
        self.slopes[1:-1] = np.where(secants[:-1]*secants[1:] <= 0, 0.,
                                     .5*(secants[:-1]+secants[1:]))

    def sample(self, now_ns):
        elapsed = (int(now_ns)-self.origin_ns)/1e9
        if not 0 <= elapsed < (len(self.q)-1)*self.dt:
            raise ValueError('prediction does not cover execution timestamp')
        i = min(int(elapsed/self.dt), len(self.q)-2)
        u = (elapsed-i*self.dt)/self.dt
        q0, q1 = self.q[i:i+2]
        v0, v1 = self.slopes[i:i+2]
        q = ((2*u**3-3*u*u+1)*q0 + (u**3-2*u*u+u)*self.dt*v0
             + (-2*u**3+3*u*u)*q1 + (u**3-u*u)*self.dt*v1)
        v = ((6*u*u-6*u)/self.dt*q0 + (3*u*u-4*u+1)*v0
             + (-6*u*u+6*u)/self.dt*q1 + (3*u*u-2*u)*v1)
        return q, v


class RollingActions:
    def __init__(self):
        self.chunks = []
        self.last_origin_ns = -1
        self.started_ns = None
        self.initial = None
        self.ever_sampled = False
        self.ramp_start_ns = None
        self.stopped = False

    def begin(self, initial8, now_ns):
        if self.started_ns is not None or self.stopped:
            raise ValueError('rolling execution cannot restart')
        q = np.asarray(initial8, dtype=float)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError('invalid rolling start reference')
        self.initial = q.copy()
        self.started_ns = int(now_ns)
        self.chunks.clear()  # Never execute a dry-run prefix from mid-chunk.

    def stop(self):
        self.stopped = True
        self.chunks.clear()

    def add(self, positions, metadata, now_ns):
        if self.stopped:
            return False
        chunk = Chunk(positions, metadata, now_ns)
        age = int(now_ns)-chunk.origin_ns
        if not 0 <= age < MAX_AGE_NS:
            raise ValueError('rolling prediction observation is stale/future')
        if chunk.origin_ns <= self.last_origin_ns:
            raise ValueError('duplicate or reordered rolling observation')
        self.last_origin_ns = chunk.origin_ns
        if self.started_ns is not None and chunk.origin_ns < self.started_ns:
            return False  # Inference begun before START is not a new episode action.
        if self.started_ns is not None and metadata.get('reset'):
            raise ValueError('unexpected policy reset during physical rolling execution')
        self.chunks = [c for c in self.chunks if int(now_ns)-c.origin_ns < MAX_AGE_NS]
        self.chunks.append(chunk)
        if self.started_ns is not None and self.ramp_start_ns is None:
            self.ramp_start_ns = int(now_ns)
        if len(self.chunks) > 32:
            raise ValueError('unexpected rolling prediction rate')
        return True

    def sample(self, now_ns):
        """Return (q, qd, provenance), or None only while awaiting START data."""
        if self.stopped:
            raise ValueError('rolling execution stopped')
        now_ns = int(now_ns)
        if self.started_ns is None:
            return None
        if now_ns < self.started_ns:
            raise ValueError('non-monotonic rolling execution clock')
        contributors = []
        remaining, remaining_dot = 1., 0.
        for c in self.chunks:
            age = (now_ns-c.origin_ns)/1e9
            if not 0 <= age < EXPIRY_NS/1e9:
                continue
            admitted, admitted_dot = smoothstep((now_ns-c.received_ns)/1e9, FADE_S)
            expired, expired_dot = smoothstep(age-(EXPIRY_NS/1e9-FADE_S), FADE_S)
            base = admitted*(1-expired)
            base_dot = admitted_dot*(1-expired)-admitted*expired_dot
            # Preserve the current fresh tail; new inference must not reset
            # execution to its leading samples. Fade the old tail out before
            # its ORIGINAL observation deadline, handing weight to newer ones.
            w = remaining*base
            wd = remaining_dot*base+remaining*base_dot
            remaining_dot = remaining_dot*(1-base)-remaining*base_dot
            remaining *= 1-base
            if w > 0:
                q, v = c.sample(now_ns)
                contributors.append((c, w, wd, q, v))
        if not contributors:
            if self.ever_sampled or now_ns-self.started_ns >= MAX_AGE_NS:
                raise ValueError('no fresh rolling prediction covers this execution tick')
            return None
        total = sum(x[1] for x in contributors)
        total_dot = sum(x[2] for x in contributors)
        q = sum(w*value for _, w, _, value, _ in contributors)/total
        v = (sum(wd*value+w*velocity for _, w, wd, value, velocity in contributors)
             - q*total_dot)/total
        # Start at the held command with zero velocity. No model-state offset.
        startup, startup_dot = smoothstep((now_ns-self.ramp_start_ns)/1e9, START_BLEND_S)
        v = startup*v+startup_dot*(q-self.initial)
        q = self.initial+startup*(q-self.initial)
        self.ever_sampled = True
        newest = max(contributors, key=lambda x: x[0].origin_ns)[0]
        oldest_ns = min(x[0].origin_ns for x in contributors)
        metadata = {**newest.metadata, 'observation_monotonic_ns': oldest_ns,
                    'source_age_ns': max(c.metadata.get('source_age_ns', 0) for c, *_ in contributors),
                    'source_skew_ns': max(c.metadata.get('source_skew_ns', 0) for c, *_ in contributors),
                    'newest_observation_monotonic_ns': newest.origin_ns,
                    'evaluated_monotonic_ns': now_ns,
                    'contributors': [{'chunk_id': c.metadata['queue_chunk_id'],
                                      'observation_monotonic_ns': c.origin_ns,
                                      'weight': w/total,
                                      'raw_step': (now_ns-c.origin_ns)/INTERVAL_NS}
                                     for c, w, _, _, _ in contributors]}
        if not np.isfinite(np.r_[q, v]).all():
            raise ValueError('nonfinite rolling q/qd')
        return q, v, metadata
