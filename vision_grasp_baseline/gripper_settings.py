"""Right pinch-gripper targets in radians (negative opens the fingers)."""
OPEN_POSITION=-0.45
CLOSE_POSITION=-0.085

# New collection contract. Do not reinterpret legacy open-transit recordings or
# ACT checkpoints; this version is written into each new plan/session/episode.
COLLECTION_POLICY='closed_transit_v2'
# Empty fingers need a different target from fingers holding the 40mm cube.
# Stay below the 0-rad hard endpoint; never increase gains or command beyond it.
TRANSIT_POSITION=-.02
# Operator confirmed physical closure at approximately -0.0494 rad on this
# gripper (cycle_0013 CAN feedback). Allow a small lower-side measurement
# margin, but never extend acceptance above the 0-rad hard position limit.
TRANSIT_MIN_POSITION=-.055
TRANSIT_MAX_POSITION=0.
TRANSIT_TOLERANCE=.035  # Maximum error; acceptance is asymmetric, see bounds.
TRANSIT_SETTLE_S=1.0
TRANSIT_STABLE_S=.30
TRANSIT_GUARD_MARGIN_S=.25
COLLECTION_EVENTS=('reach','pregrasp_open','approach','close','carry','open',
                   'return','transit_close','return_closed')


def require_closed_transit(measured, command=None):
    import math
    if not math.isfinite(measured) or not TRANSIT_MIN_POSITION<=measured<=TRANSIT_MAX_POSITION:
        raise ValueError('Closed-transit collection requires an empty closed right gripper')
    if command is not None and (not math.isfinite(command) or abs(command-TRANSIT_POSITION)>.005):
        raise ValueError('Right gripper controller reference is not the closed-transit target')


def require_current_plan(plan):
    """Fail before replay/recording; never silently upgrade a cached trajectory."""
    if plan.get('gripper_policy')!=COLLECTION_POLICY:
        raise ValueError('Legacy/missing gripper policy; replan for '+COLLECTION_POLICY)
    events=plan.get('events',[])
    if tuple(e.get('name') for e in events)!=COLLECTION_EVENTS:
        raise ValueError('Closed-transit event sequence missing/changed')
    times=[float(e['time_s']) for e in events]
    import math
    if not all(math.isfinite(t) for t in times) or times[0]!=0 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Invalid closed-transit event times')
    samples=plan.get('samples',[])
    if not samples or times[-1]>=samples[-1]['time_s']:
        raise ValueError('Closed return phase missing')
    stamps=[float(r['time_s']) for r in samples]
    if (not all(math.isfinite(t) for t in stamps) or stamps[0]!=0
            or any(b<=a for a,b in zip(stamps,stamps[1:]))):
        raise ValueError('Invalid closed-transit sample times')
    for row in samples:
        if any(len(row[k])!=8 or not all(math.isfinite(x) for x in row[k])
               for k in ('positions','velocities','accelerations')):
            raise ValueError('Invalid closed-transit sample vectors')
    for row in (samples[0],samples[-1]):
        if len(row['positions'])!=8 or abs(row['positions'][7]-TRANSIT_POSITION)>1e-8:
            raise ValueError('Closed-transit plan must start/end with closed gripper')
    # The synchronized controller uses quintic interpolation. Equal arm
    # endpoints alone are insufficient: nonzero derivatives could move the arm
    # between them while opening/closing at a verified stationary point.
    for start,end,target in ((1,2,OPEN_POSITION),(3,4,CLOSE_POSITION),
                             (5,6,OPEN_POSITION),(7,8,TRANSIT_POSITION)):
        rows=[r for r in samples if times[start]-1e-9<=r['time_s']<=times[end]+1e-9]
        if (len(rows)<3 or abs(rows[0]['time_s']-times[start])>1e-9
                or abs(rows[-1]['time_s']-times[end])>1e-9):
            raise ValueError('Missing gripper transition boundary samples')
        reference=rows[0]['positions'][:7]
        if any(abs(r['positions'][i]-reference[i])>1e-8
               or abs(r['velocities'][i])>1e-8 or abs(r['accelerations'][i])>1e-8
               for r in rows for i in range(7)):
            raise ValueError('Arm must remain stationary during gripper transition')
        if abs(rows[-1]['positions'][7]-target)>1e-8:
            raise ValueError('Incorrect gripper transition target')
        if start==7 and rows[-1]['time_s']-rows[-2]['time_s']<TRANSIT_SETTLE_S-1e-9:
            raise ValueError('Missing empty-gripper settling interval before return')
        if start==7 and any(abs(r['positions'][7]-target)>1e-8
                            or abs(r['velocities'][7])>1e-8 or abs(r['accelerations'][7])>1e-8
                            for r in rows[-2:]):
            raise ValueError('Empty-gripper settling interval must hold its closed target')
    for index,target in ((0,TRANSIT_POSITION),(2,OPEN_POSITION),(4,CLOSE_POSITION),
                         (6,OPEN_POSITION),(8,TRANSIT_POSITION)):
        end=times[index+1] if index<8 else stamps[-1]
        rows=[r for r in samples if times[index]-1e-9<=r['time_s']<=end+1e-9]
        if any(abs(r['positions'][7]-target)>1e-8 or abs(r['velocities'][7])>1e-8
               or abs(r['accelerations'][7])>1e-8 for r in rows):
            raise ValueError('Incorrect gripper hold during arm travel')
