"""Try fresh complete candidates before recording or dispatching any motion."""
from pathlib import Path

from .perception_retry import StageFailure


def plan_complete(run,out,scene,registration,model,attempt_log):
    for attempt in range(3):
        directory=Path(out)/('complete_plan' if attempt==0 else f'complete_plan_retry_{attempt}')
        try:
            run('continuous-plan','--scene',scene,'--registration',registration,
                '--robot-model',model,'--output',directory)
        except StageFailure as exc:
            if exc.stage!='continuous-plan' or exc.returncode!=1:raise
            attempt_log.append(dict(output=str(directory),accepted=False,error=str(exc)))
            if attempt==2:
                raise RuntimeError('3 complete trajectory candidates failed before execution; '
                                   'no grasp dispatched. Inspect complete_plan*/result.json') from exc
            print(f'完整轨迹候选未通过，重新规划（{attempt+2}/3）；尚未开始抓取。',flush=True)
            continue
        attempt_log.append(dict(output=str(directory),accepted=True))
        return directory/'trajectory.json'
