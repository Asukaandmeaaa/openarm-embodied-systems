"""Bounded, pre-motion recapture; rejected batches never become motion input."""
from pathlib import Path


class StageFailure(RuntimeError):
    def __init__(self,stage,returncode,log_path,tail):
        self.stage=stage;self.returncode=returncode;self.log_path=log_path
        super().__init__(f'{stage} failed (exit {returncode}). Log: {log_path}\n{tail}')


def capture_stable_scene(run,out,name,calibration,attempt_log,*,stationary=False):
    for attempt in range(3):
        directory=Path(out)/(name if attempt==0 else f'{name}_retry_{attempt}')
        samples=10 if attempt==0 else 20
        run('capture','--output',directory,'--samples',samples)
        try:
            flags=['--stationary-fixture'] if stationary else []
            run('detect','--capture',directory,'--calibration',calibration,*flags)
        except StageFailure as exc:
            if exc.stage!='detect' or exc.returncode!=3:raise
            attempt_log.append(dict(capture=str(directory),samples=samples,accepted=False,error=str(exc)))
            if attempt==2:
                raise RuntimeError(f'{name}: 3 fresh image batches remained unstable; '
                                   'no trajectory dispatched. Inspect scene_unstable.json in capture directories') from exc
            print(f'识别暂不稳定，自动重新采集（{attempt+2}/3，20帧）；保持方块和盒子不动。',flush=True)
            continue
        attempt_log.append(dict(capture=str(directory),samples=samples,accepted=True))
        return directory/'scene.json'
