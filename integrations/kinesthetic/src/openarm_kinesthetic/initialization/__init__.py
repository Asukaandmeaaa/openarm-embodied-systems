"""Initialization planning: prior table geometry -> teaching-ready -> planner -> handoff.

Simulation and hardware share everything in this package. Only the geometry provider
differs (`SimTableGeometryProvider` vs `RealTableGeometryProvider`, both in
`table_geometry`), which is the whole architectural point: migrating to the real
station must not mean rewriting IK, planning, the state machine or the handoff.
"""
