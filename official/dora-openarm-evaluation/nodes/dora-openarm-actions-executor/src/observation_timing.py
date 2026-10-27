"""Small dependency-free timing contract shared by Dora validation nodes.

The three timing values have different meanings and must not be conflated:

``source_age_ns`` and ``source_skew_ns`` are measured by the observer when
the camera/feedback sample is assembled.  ``observation_monotonic_ns`` is the
monotonic timestamp of that assembled observation.  Downstream nodes may
validate the current observation age, but they must preserve the two source
measurements when forwarding the metadata.
"""


def validate_timing(
    metadata,
    now_ns,
    *,
    max_source_age_ns=200_000_000,
    max_source_skew_ns=100_000_000,
    max_observation_age_ns=500_000_000,
    required=True,
):
    """Validate source freshness, synchronization, and observation age."""
    required_keys = ("observation_monotonic_ns", "source_age_ns", "source_skew_ns")
    missing = [key for key in required_keys if key not in metadata]
    if missing and required:
        raise ValueError(f"missing source timing metadata: {','.join(missing)}")
    if missing:
        return None
    observation_ns = int(metadata["observation_monotonic_ns"])
    age_ns = int(now_ns) - observation_ns
    source_age_ns = int(metadata["source_age_ns"])
    source_skew_ns = int(metadata["source_skew_ns"])
    if age_ns < 0:
        raise ValueError("observation timestamp is from the future")
    if source_age_ns < 0:
        raise ValueError(f"source age is negative: age_ns={source_age_ns}")
    if source_skew_ns < 0:
        raise ValueError(f"source skew is negative: skew_ns={source_skew_ns}")
    if source_age_ns > max_source_age_ns:
        raise ValueError(f"source data stale: age_ns={source_age_ns}")
    if source_skew_ns > max_source_skew_ns:
        raise ValueError(f"source data unsynchronized: skew_ns={source_skew_ns}")
    if age_ns > max_observation_age_ns:
        raise ValueError(f"observation stale: age_ns={age_ns}")
    return {
        "observation_monotonic_ns": observation_ns,
        # Keep source measurements from the observer.  ``age_ns`` is a
        # downstream transport/queue delay and belongs to the observation-age
        # check, not to the source-age field.
        "source_age_ns": source_age_ns,
        "source_skew_ns": source_skew_ns,
    }


def observation_deadline_ns(metadata, max_observation_age_ns=500_000_000):
    """Return the fail-closed deadline for a running action chunk.

    The deadline is derived from the original observation timestamp.  It must
    not be recomputed from the time an action chunk reaches the executor: a
    slow policy response must shorten the remaining action horizon.
    """
    return int(metadata["observation_monotonic_ns"]) + int(max_observation_age_ns)
