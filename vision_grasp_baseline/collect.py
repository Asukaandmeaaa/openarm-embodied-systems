"""Attended fixed-scene repetitions: zero -> record -> pick/place -> zero -> save."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .collection_io import rpc, write_json
from .gripper_settings import COLLECTION_POLICY,require_current_plan
from .pickup_clearance import require_pickup_clearance
from .collection_layout import create_layout,next_episode_number,mark_failed_attempt

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
COLLECTION_TYPES = {
    'nearby': '常用位置附近，小幅平移',
    'spread': '已验证可抓取范围内，前后左右分散摆放',
    'rotation': '已验证可抓取范围内，增加小角度旋转',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=30, help='Number of successful, operator-reviewed episodes')
    parser.add_argument('--output', type=Path,help='Optional NEW isolated session directory; default appends to classified collections')
    parser.add_argument('--tag', choices=COLLECTION_TYPES,default='nearby',
                        help='Category: nearby (default), spread, rotation; controls default storage directory')
    parser.add_argument('--staged',action='store_true',help='Use the legacy slow, individually planned execution')
    parser.add_argument('--replay-plan',type=Path,help='Reuse one saved complete trajectory; fresh 5mm fixture gate each episode')
    parser.add_argument('--camera-check', action='store_true', help='Only benchmark both cameras and save a fresh RGB-D snapshot; no robot control')
    parser.add_argument('--seconds', type=float, default=10., help='Camera benchmark duration')
    args = parser.parse_args()
    if args.staged:parser.error('Closed-transit collection requires the complete continuous planner; --staged is legacy')
    if args.replay_plan:
        cached=json.loads(args.replay_plan.read_text())
        require_current_plan(cached);require_pickup_clearance(cached)
    if args.episodes < 1 or args.seconds < 2:
        parser.error('episodes must be positive and seconds >= 2')
    lock = (BASE/'.collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    layout=create_layout(BASE,args.tag,args.output,camera_only=args.camera_check)
    out=layout.session
    if shutil.disk_usage(out).free < 10*1024**3:
        raise RuntimeError('At least 10 GiB free disk space required')
    manifest = dict(schema_version=3,requested_episodes=args.episodes, completed_episodes=[],
                    attempted_episodes=[],camera_only=args.camera_check,
                    storage_layout='classified' if layout.classified else 'isolated_session',
                    session_directory=str(out),episodes_directory=str(layout.episodes),runs_directory=str(layout.runs),
                    gripper_policy=COLLECTION_POLICY,
                    stationary_fixture=True,
                    collection_tag=args.tag, collection_description=COLLECTION_TYPES.get(args.tag),
                    started_wall_ns=time.time_ns(), success=False)
    service = None
    active_attempt = None
    try:
        if not args.camera_check:
            # Fail before opening cameras when the arm backend is absent/not ready.
            from .cycle import wait_for_gateway_handoff
            gateway_state=wait_for_gateway_handoff(synchronize_if_needed=True)
            manifest['controller_command_interfaces']=(gateway_state or {}).get('controller_command_interfaces',{})
        ready = out/'service.json'
        cmd = [str(BASE/'run.sh'), 'collection-service', '--ready', str(ready)]
        if args.camera_check:
            cmd.extend(['--camera-only', '--benchmark-dir', str(out/'camera_frames')])
        with (out/'service.log').open('w') as log:
            service = subprocess.Popen(cmd, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30.
        error = 'waiting for service startup'
        while True:
            if service.poll() is not None:
                raise RuntimeError('Capture service exited; inspect ' + str(out/'service.log'))
            if ready.exists():
                settings = json.loads(ready.read_text())
                os.environ['VISION_CAPTURE_SERVICE'] = settings['url']
                os.environ['VISION_CAPTURE_TOKEN'] = settings['token']
                try:
                    health = rpc('status')
                    break
                except Exception as exc:
                    error = str(exc)
            if time.monotonic() >= deadline:
                raise RuntimeError('Capture startup failed: '+error)
            time.sleep(.2)
        write_json(out/'preflight.json', health)
        print('相机/记录器已就绪；本轮会话日志：'+str(out), flush=True)
        if not args.camera_check:
            print('数据目录：'+str(layout.episodes)+'；执行记录：'+str(layout.runs),flush=True)
        if args.tag:
            print(f'采集类型：{args.tag} — {COLLECTION_TYPES[args.tag]}', flush=True)
        if not args.camera_check:
            print('执行模式：'+('旧分段流程' if args.staged else '完整轨迹缓存回放' if args.replay_plan else '每轮完整预规划、一次连续执行'),flush=True)
            print(f'夹爪流程 {COLLECTION_POLICY}：空爪闭合零位 → 预抓取点开爪 → 抓放 → 撤离后闭爪并验收 → 回零。',flush=True)
        if args.camera_check:
            deadline = time.monotonic()+args.seconds
            health_samples = []
            while time.monotonic() < deadline:
                health_samples.append(rpc('status'))
                time.sleep(.2)
            rpc('capture', output=str(out/'snapshot'), samples=10)
            write_json(out/'camera_check.json', dict(physical_motion=False, checks=health_samples))
            print(json.dumps(health_samples[-1], indent=2), flush=True)
        else:
            # --episodes counts accepted demonstrations, not attempts. A
            # rejected, normally completed cycle must not consume this quota.
            while len(manifest['completed_episodes']) < args.episodes:
                if shutil.disk_usage(out).free < 10*1024**3:
                    raise RuntimeError('Free disk space below 10 GiB; collection stopped')
                placement='盒外缓存轨迹的原位置（偏移不超过5 mm）' if args.replay_plan else '盒外抓取区域（相对本次首轮位置偏移不超过4 cm）'
                while True:
                    response = input(f'[合格 {len(manifest["completed_episodes"])}/{args.episodes}，'
                        f'准备第 {len(manifest["attempted_episodes"])+1} 次] '
                        f'把方块放到{placement}、保持盒子不动，手离开工作区。回车开始本轮；q 退出：').strip().lower()
                    if response in ('', 'q'):
                        break
                    print('尚未开始本轮：请回车开始，或输入 q 退出。',flush=True)
                if response == 'q':
                    manifest['exit_reason']='operator_quit_before_next_attempt'
                    break
                rpc('status')
                number=next_episode_number(layout)
                episode=layout.episodes/f'episode_{number:03d}'
                run=layout.runs/f'cycle_{number:03d}'
                active_attempt = dict(number=number,episode=str(episode),run=str(run),status='started')
                manifest['attempted_episodes'].append(active_attempt)
                write_json(out/'session.json',manifest)
                # Camera environment propagates into all perception/verification children.
                mode=['--waypoints'] if args.staged else ['--continuous']
                mode.append('--stationary-fixture')
                if args.replay_plan:mode.extend(['--replay-plan',str(args.replay_plan.resolve())])
                subprocess.run([str(BASE/'run.sh'), 'cycle', *mode, '--output', str(run),
                                '--record-episode', str(episode), '--scene-reference', str(out/'fixed_scene.json')],
                               cwd=WORKSPACE, check=True)
                result = json.loads((run/'cycle_result.json').read_text())
                if not result['success']:
                    raise RuntimeError('Cycle failed: '+str(run))
                metadata = json.loads((episode/'metadata.json').read_text())
                metadata['collection_tag'] = args.tag
                metadata['gripper_policy']=COLLECTION_POLICY
                metadata['collection_description'] = COLLECTION_TYPES.get(args.tag)
                metadata['controller_command_interfaces']=manifest.get('controller_command_interfaces',{})
                metadata['cycle_result'] = str(run/'cycle_result.json')
                metadata['collection_session']=str(out/'session.json')
                metadata['source_trajectory_trial_id'] = out.name+'_'+run.name
                while True:
                    review = input('本轮已回零并保存。确认抓取、入盒、松爪均成功？'
                        'y 合格并继续；n 不合格、保留待审核并继续补采；q 保留待审核并退出：').strip().lower()
                    if review in ('y','n','q'):
                        break
                    print('请输入 y、n 或 q；空输入不会确认合格，也不会开始下一轮。',flush=True)
                accepted = review == 'y'
                metadata['review_status'] = 'approved' if accepted else 'pending'
                metadata['operator_marked_success'] = accepted
                metadata['operator_review_response'] = review
                write_json(episode/'metadata.json', metadata)
                if not accepted:
                    active_attempt['status'] = 'pending_review'
                    active_attempt['operator_review_response'] = review
                    write_json(episode/'REVIEW_REQUIRED.json', dict(
                        reason='Operator did not approve this episode', operator_response=review,
                        eligible_for_training=False))
                    active_attempt = None
                    if review == 'q':
                        manifest['exit_reason']='operator_quit_at_review'
                    write_json(out/'session.json',manifest)
                    print('本轮保留待审核，不计入合格数量。'+
                          ('请重新摆放方块，下一轮仍需回车才开始。' if review=='n' else '本次采集结束。'),flush=True)
                    if review == 'q':
                        break
                    continue
                active_attempt['status'] = 'approved'
                active_attempt = None
                manifest['completed_episodes'].append(str(episode))
                write_json(out/'session.json', manifest)
                print(f'已保存合格数据 {len(manifest["completed_episodes"])}/{args.episodes}', flush=True)
        manifest['success'] = args.camera_check or len(manifest['completed_episodes']) == args.episodes
    except BaseException as exc:
        manifest['error'] = str(exc) or type(exc).__name__
        manifest['interrupted'] = isinstance(exc, (KeyboardInterrupt, EOFError))
        if not args.camera_check:
            (BASE/'STOP').touch()
        raise
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=15.)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()
                manifest['service_shutdown_error'] = 'Timed out; inspect partial episode before reuse'
        if active_attempt is not None:
            try:
                marker=mark_failed_attempt(layout,active_attempt,
                    reason=manifest.get('error','Collection exited before approval'),
                    interrupted=manifest.get('interrupted',False))
                print('本轮失败/中断标记：'+str(marker),flush=True)
            except Exception as exc:
                manifest['failure_marker_error']=str(exc)
        if manifest.get('error'):
            label='INTERRUPTED' if manifest.get('interrupted') else 'FAILED'
            detail=dict(reason=manifest['error'],session=str(out/'session.json'),
                        eligible_for_training=False,attempt=active_attempt)
            write_json(out/(label+'.json'),detail)
            if active_attempt is None:
                # Startup/status/input failures may occur before any episode
                # exists. Still expose a visible, non-training failure folder.
                directory=layout.runs.parent/'failed'/(out.name+'_'+label)
                directory.mkdir(parents=True,exist_ok=False)
                write_json(directory/(label+'.json'),detail)
                manifest['failure_marker']=str(directory/(label+'.json'))
        manifest['finished_wall_ns'] = time.time_ns()
        write_json(out/'session.json', manifest)
        # Also label interrupted/pending recordings after the recorder has stopped.
        # Preserve their quality and approval status unchanged.
        # Data may live in shared category directories. Only label this
        # invocation's attempts, never all older episodes in the category.
        for attempt in manifest['attempted_episodes']:
            metadata_path=Path(attempt['episode'])/'metadata.json'
            if not metadata_path.exists():continue
            metadata = json.loads(metadata_path.read_text())
            metadata['collection_tag'] = args.tag
            metadata['collection_description'] = COLLECTION_TYPES.get(args.tag)
            metadata['controller_command_interfaces']=manifest.get('controller_command_interfaces',{})
            metadata['collection_session']=str(out/'session.json')
            metadata['source_trajectory_trial_id']=out.name+'_'+Path(attempt['run']).name
            if attempt.get('failure_marker'):
                metadata.update(collection_failure_marker=attempt['failure_marker'],
                    review_status='pending',operator_marked_success=False,diagnostic_only=True)
            write_json(metadata_path, metadata)
        lock.close()


if __name__ == '__main__':
    main()
