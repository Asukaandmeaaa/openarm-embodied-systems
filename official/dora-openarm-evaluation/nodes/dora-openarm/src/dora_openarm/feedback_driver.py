"""Dora driver with bounded explicit CAN feedback requests."""

import time

from openarm_driver import SingleArmDriver as BaseSingleArmDriver
from openarm_driver.config import get_default_config

from .can_runtime import require_paced_can
from .can_ownership import CanOwnership


class SingleArmDriver(BaseSingleArmDriver):
    """Poll received data on every read; request new data at most at 20 Hz.

    Eight paced refresh frames take about 8 ms. Dora timer events can arrive
    faster, so they must not each trigger another refresh batch. Motion
    commands and their receive processing remain in the underlying driver.
    """

    def __init__(self, *args, **kwargs):
        require_paced_can()
        arm_side = args[0] if args else kwargs["arm_side"]
        config = args[1] if len(args) > 1 else kwargs.get("config")
        config = get_default_config() if config is None else config
        interface = args[5] if len(args) > 5 else kwargs.get("can_interface")
        interface = config.get_can_interface(arm_side) if interface is None else interface
        self._can_ownership = CanOwnership(interface, f"dora-direct:{arm_side}")
        self._next_can_refresh = 0.0
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._close_transport()
            raise

    def _close_transport(self):
        """Drop SDK/socket references before releasing ownership; sends no CAN."""
        self._health_collections = []
        self.openarm = None
        owner, self._can_ownership = getattr(self, "_can_ownership", None), None
        if owner is not None:
            owner.close()

    def stop(self):
        """Retain ownership through the existing disable/settle sequence."""
        super().stop()
        # If stop raises, retain ownership rather than admit another writer.
        self._close_transport()

    def __del__(self):
        # Destruction never enables, disables or moves an arm. Normal stop()
        # remains responsible for the pre-existing shutdown behavior.
        self._close_transport()

    def fetch_state(self, refresh=True):
        """Keep receiving each call without flooding the query transmit queue."""
        now = time.monotonic()
        request = refresh and now >= self._next_can_refresh
        state = super().fetch_state(refresh=request)
        if request:
            self._next_can_refresh = now + 0.05
        return state

    def observation_metadata(self):
        """Return a source stamp derived from the oldest fresh axis reply.

        The observer must not make a cached CAN state look current merely
        because it was forwarded later.  ``seconds_since_response`` is based
        on the driver's steady clock, so the stamp is in the same monotonic
        domain as the camera capture nodes.
        """
        collection = self.openarm.get_arm()
        ages = [float(collection.get_link_stats(i).seconds_since_response())
                for i in range(8)]
        if not ages or not all(age >= 0.0 for age in ages):
            raise RuntimeError("invalid CAN feedback age")
        if any(age == float("inf") for age in ages):
            raise RuntimeError("CAN feedback has not replied on every axis")
        return {"source_monotonic_ns": time.monotonic_ns() - int(max(ages) * 1e9)}
