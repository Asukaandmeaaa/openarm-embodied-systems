"""Line-based operator console. No hardware is touched until an explicit command."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT/'scripts/actctl'
HELP = '''键盘控制（按键后回车；不是实体急停）
  s  检查 ready，通过后请求 START
  x / p  停止/暂停 ACT，保留 ROS 保持；不支持任意姿态续跑
  h  只规划回 HOME（先退出 policy/Dora，gateway 必须 idle，夹爪已打开）
  r  执行已检查的 HOME 规划（另需输入 HOME 确认，会运动）
  b  恢复 HOME 向导：退出本轮 policy/Dora、检查故障、规划并确认执行
  t  查看状态
  q  请求 ACT 停止并退出键盘；不关闭 H1/gateway
H1/gateway/H2、policy/Dora 仍按各自终端启动。b 经确认可复位已排空的候选碰撞故障；不打开夹爪或重启硬件。
回 HOME 执行中 Ctrl+C 用于取消该回 HOME 命令；危险时使用实体急停。
'''


def call(stage):
    return subprocess.run(['bash', str(CLI), stage], cwd=ROOT, check=False).returncode


def dispatch(key, run=call, confirm=input):
    if key == 's':
        if run('ready') == 0:
            run('start')
            run('status')
    elif key in ('x', 'p'):
        run('stop')
        run('status')
    elif key == 'h':
        run('plan-home')
    elif key == 'b':
        run('home')
    elif key == 'r':
        if confirm('确认已检查 plan.json、夹爪空载且路径安全；输入 HOME 执行：') == 'HOME':
            run('return-home')
    elif key == 't':
        run('status')
    elif key == 'q':
        run('stop')
        return False
    else:
        print(HELP)
    return True


def main():
    print(HELP)
    while True:
        try:
            if not dispatch(input('ACT> ').strip().lower()):
                return
        except EOFError:
            call('stop')
            return
        except KeyboardInterrupt:
            # SIGINT also reaches a foreground return-home child, cancelling
            # its lease. actctl stop only stops ACT, not an independent H2.
            print('\n键盘中断：请求 ACT 停止。保留 H1/gateway；请检查状态。')
            call('stop')
            return


if __name__ == '__main__':
    main()
