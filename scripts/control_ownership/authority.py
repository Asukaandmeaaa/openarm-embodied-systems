"""Pure lease state machine. Caller serializes operations and certifies draining.

An expired connection never transfers ownership automatically: the motor backend
must finish cancelling/holding first. Tokens are capabilities, not log fields.
"""
import math
import secrets


class LeaseAuthority:
    def __init__(self, ttl=.5):
        if not math.isfinite(ttl) or not .1 <= ttl <= 2.:
            raise ValueError('invalid lease TTL')
        self.ttl = ttl
        self.owner = self.session = self.token = None
        self.expires_at = None
        self.state = 'idle'
        self.reason = None
        self.last_time = None

    def _time(self, now):
        if not math.isfinite(now) or now < 0 or (self.last_time is not None and now < self.last_time):
            raise ValueError('invalid/non-monotonic lease clock')
        self.last_time = now
        if self.state == 'owned' and now >= self.expires_at:
            self.state = 'draining'
            self.reason = 'lease_expired'
            self.token = None

    def acquire(self, owner, session, now):
        self._time(now)
        if owner not in ('h2', 'dora') or not isinstance(session, str) or not session:
            raise ValueError('invalid owner/session')
        if self.state != 'idle':
            raise ValueError('command authority is '+self.state)
        self.owner, self.session = owner, session
        self.token = secrets.token_hex(32)
        self.expires_at = now + self.ttl
        self.state, self.reason = 'owned', None
        return self.token

    def _check(self, token, session, now):
        self._time(now)
        if (self.state != 'owned' or not isinstance(token, str)
                or not secrets.compare_digest(token, self.token or '') or session != self.session):
            raise ValueError('no live authority for this connection')

    def renew(self, token, session, now):
        self._check(token, session, now)
        self.expires_at = now + self.ttl

    def authorize(self, token, session, now, operation):
        self._check(token, session, now)
        if {'command': 'dora', 'action': 'h2'}.get(operation) != self.owner:
            raise ValueError('operation not authorized for owner')
        return True

    def release(self, token, session, now):
        self._check(token, session, now)
        self.token = None
        self.state, self.reason = 'draining', 'released'

    def disconnect(self, session, now):
        self._time(now)
        if self.session == session and self.state == 'owned':
            self.token = None
            self.state, self.reason = 'draining', 'connection_closed'

    def drain_completed(self, now):
        self._time(now)
        if self.state != 'draining':
            raise ValueError('not draining')
        self.owner = self.session = self.token = self.expires_at = None
        self.state, self.reason = 'idle', None

    def fault(self, reason):
        self.token = None
        self.state, self.reason = 'fault', str(reason)

    def reset_fault(self, now):
        self._time(now)
        if self.state != 'fault':
            raise ValueError('not faulted')
        self.owner = self.session = self.token = self.expires_at = None
        self.state, self.reason = 'idle', None

    def snapshot(self, now):
        self._time(now)
        return {'state': self.state, 'owner': self.owner, 'expires_at': self.expires_at,
                'fault': self.reason if self.state == 'fault' else None,
                'reason': self.reason, 'draining': self.state == 'draining',
                'busy': self.state != 'idle'}
