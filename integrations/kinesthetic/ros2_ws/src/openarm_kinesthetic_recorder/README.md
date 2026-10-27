# openarm_kinesthetic_recorder

当前 ROS 2 包只安装已验证的固定 HOME、真机控制、录制和回放入口：

- `home_guarded_executor`：fresh-plan 并执行 zero→side-prehome→fixed HOME，然后张开 J8 并验收。
- `real_gamepad`：G3V2 输入、控制模式、录制 toggle 和 continuous_marked 语义事件。
- `right_kinesthetic_bridge`：J1–J7 刚性/passive 切换、受限笛卡尔微步和独立夹爪命令。
- `right_motion_recorder`：默认连续采集完整 J1–J8 帧并保存 schema v2 marker 轨迹；旧 continuous/keyframe 兼容。
- `recorded_motion_player`：H2 后 dense-path PLAN ONLY、录制时序语义分段回放和 runtime return HOME；旧 v1 兼容。
- `wrist_camera`、`real_episode_recorder`：可选完整相机 episode 采集。
- `teaching_status`：运行状态摘要。
- `act_executor`：显式启用后读取 Nano `/dev/shm/act_live/latest_action.json`，只向右 J1–J7/J8
  的 FollowJointTrajectory 接口发送经过校验的 10 步、30 Hz ACT chunk。默认禁用；启动
  前应停止 bridge、recorder、player 等其他运动源；gamepad 保持连接且 idle，用于 RT/RB 停止。

ACT 执行器不会因为动作文件存在而自动运动。真机启动需要同时显式设置
`allow_real_execution:=true` 和 `enabled:=true`；也可先以 `enabled:=false` 启动，确认
`/joint_states` 和两个控制器就绪后调用 `ros2 service call /act_executor/start std_srvs/srv/Trigger {}`。
`/act_executor/stop` 会提交右臂与右夹爪当前 goal 的异步取消请求；服务返回尚不表示
控制器已经确认取消。文件 schema、递增动作/观测序号、单调
时钟有效期、完整 16 维动作与右侧 8 维投影、30 Hz 和当前 joint state 都是硬门槛；
失败会锁存并停止继续下发，同时提交取消请求，需重启节点后再试。

2026-10-07 真机 ACT 已产生连续 FJT goal 并抓住方块，但出现抖动和抓取/放置阶段停滞。
当前 FJT 10 点轨迹长约 0.32 s，现场 chunk 间隔约 0.46–0.48 s；实际 reference/feedback
尚未同步记录，不能归因于模型或控制器。下一步诊断见
`docs/ACT_NANO_REAL_EXECUTION_2026-10-07.md`。

旧 table initializer、ready-profile、旧 guarded executor 和旧 bringup 不再安装。
完整现场命令见仓库 `docs/HOME_TEACHING_WORKFLOW.md`。
下一阶段双相机和方块位置泛化采集设计见仓库
`docs/ACT_REAL_DATA_COLLECTION_PLAN.md`。

默认 `continuous_marked` 操作：录制期间 A/Y 只产生 `semantic_only` close/open marker，
不驱动夹爪；独立 LB 不记点。播放器沿完整连续路径运动，并在 marker 处实际执行夹爪。
H2 保证起始夹爪为 OPEN；抓取位按 Y 记 CLOSE，放置位按 A 记 OPEN。
`continuous_marked` 回放保留录制样本时间戳；`--playback-rate 1.0` 为录制原速，marker
位置才实际执行夹爪。MoveIt velocity/acceleration scaling 仅用于 return HOME 和旧 keyframe。
