"""Extra planning-only cube clearance during free-space closed transit.

Nominal cube, intended contact geometry and live collision rules are unchanged.
The final smoothed controller path must pass the same inflated-obstacle check.
"""
from copy import deepcopy
import math

PICKUP_CLEARANCE_M=.030


def padded_cube(cube,margin=PICKUP_CLEARANCE_M):
    if not math.isfinite(margin) or margin<PICKUP_CLEARANCE_M:
        raise ValueError('Closed pickup requires at least 30mm planning clearance')
    result=deepcopy(cube)
    if len(result.primitives)!=1 or result.primitives[0].type!=result.primitives[0].BOX:
        raise ValueError('Pickup clearance requires the nominal cube box')
    result.primitives[0].dimensions=[float(x+2*margin) for x in result.primitives[0].dimensions]
    return result


def require_pickup_clearance(plan):
    margin=plan.get('pickup_clearance_m',0.)
    if not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<PICKUP_CLEARANCE_M:
        raise ValueError('Old trajectory lacks 30mm closed-pickup clearance; regenerate complete plan')
    return margin
