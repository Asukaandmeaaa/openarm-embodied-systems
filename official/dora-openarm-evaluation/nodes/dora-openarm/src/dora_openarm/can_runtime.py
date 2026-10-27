"""Require the locally validated CAN refresh pacing before opening hardware."""

EXPECTED_CAN_VERSION = "1.4.0+dmrefresh1"


def require_paced_can(backend=None):
    """Reject an unpatched environment without opening CAN or enabling motors."""
    if backend is None:
        import openarm_can as backend
    actual = getattr(backend, "__version__", "unknown")
    if actual != EXPECTED_CAN_VERSION:
        raise RuntimeError(
            f"Dora requires openarm_can {EXPECTED_CAN_VERSION}; found {actual}. "
            "Run openarm_vla_ws/scripts/install_dora_can_fix.sh with this "
            "Dora environment's Python path. No motor connection was opened."
        )
    return actual
