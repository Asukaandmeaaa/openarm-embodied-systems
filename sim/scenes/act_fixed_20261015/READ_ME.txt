2026-10-16 最新：目标改为300条，并加入桌面平移/水平偏转及更宽方块朝向变化。
以下命令生成新数据集，旧100条版本目录不能混写或续写为本版本。

固定场景预览，2026-10-15

文件：
measurements.json：用户测量、测量基准、未测参数和后续计划。
user_notes.txt：本次原始描述及澄清备份；原图没有本地文件，未声称备份图片。
source_snapshot/：5份已有描述快照及原路径/SHA256；仅是出处备份，不是独立可加载的资产包。
scene.xml：可直接加载的MJCF，相对引用本工程official/openarm_mujoco里的机械臂网格。
derived_geometry.json：换算坐标和XML校验和。
validation.json：编译后尺寸检查、初始碰撞检查、2秒方块落稳检查。
previews/scene_review.png：整体、俯视、侧视和尺寸图。

预览（仅本机仿真，不启动ROS/CAN）：
cd $HOME/openarm_vla_ws
.venv-real-act/bin/python scripts/build_act_fixed_scene.py --view

重新生成图片：
MUJOCO_GL=egl .venv-real-act/bin/python scripts/build_act_fixed_scene.py --render

交互窗口是静态检查，可旋转缩放；不会执行机械臂任务。
若使用MuJoCo原生加载器，请选择mechanical_zero_preview关键帧。
physical_home_reference包含现有右臂HOME和左臂零位，仅为仿真参考，不代表硬件姿态或安全许可。

本版解释：
+x机器人向前，+y机器人左侧，+z向上；z=0为立柱底板基准，不是房间地面。
立柱杆体在z=0.27m截面正面x=0.03m；桌面近边x=0.185m。
桌面左右60cm、前后40cm，中心(0.385,0)，表面z=0.27m。
方块中心(0.332,-0.196,0.290)m；靠近机器人和机器人右侧。
盒底中心(0.385,0.010,0.270)m；24cm量自机器人右侧短边到盒子外壁。
黑盒沿前后方向居中。用户已看图认可大致布局。
黑盒外14x14x4cm、内12x12x3cm、底厚1cm，沿用已有描述，并非本次实测。
桌面厚1.8cm和圆角2.5cm只是预览假设，桌腿缺少尺寸而省略。
模型保留双臂碰撞、执行器及自由方块；没有焊接抓取、运动穿透或关节瞬移示范。
采集入口 scripts/collect_act_fixed_scene.py 已接入；保留orange_cube/black_frame旧对象名。
使用接触动力学完成抓取，先验证原轨迹和30Hz命令独立回放，再写入成功示范。
图像与动作/观测均按30Hz保存，图像640x360，状态顺序为右8轴、左8轴。
机器人采用理想重力补偿；物体质量/摩擦沿用旧模型假设，尚未完成实物辨识。

当前验证与限制：
results/act_fixed_scene_pilot_20261015/contact_03：完整任务成功，约56.4秒。
results/act_fixed_scene_pilot_20261015/jitter_01：扰动摆位成功，约62.0秒。
results/act_fixed_scene_pilot_20261015/replay_pilot_01：另一摆位成功，约55.1秒；30Hz命令独立动力学回放通过。
contact_02曾提前丢块入盒，已在reevaluation.json判为失败，不算成功示范。
这些少量检查不能证明300条采集成功率；正式批量采集尚未启动。
用户确认仿真和实物ACT task均取消60秒截止：仿真按任务阶段完成或故障结束；实物手动停止或故障停止。
commissioning短检查仍为3秒。取消时间截止不改变速度/加速度/碰撞等现有规则。

全局相机：用户提供的Nano .calib已备份并应用，见source_snapshot/global_camera_20261015.calib。
该标定是world到global_camera_link；实际彩色内参与link到彩色光学中心的偏移仍待核对。
局部相机按用户意见先用官方安装参数。previews/camera_comparison.png为当前真实/仿真对照。
后续若在已安装RealSense工具的主机补齐参数，可执行以下只读查询（当前不是辅助仿真采集前提）：
rs-enumerate-devices -c > /tmp/openarm_realsense_calibration.txt
需要Color 640x360内参、设备型号/序列号及Depth到Color外参；此命令不启动机器人。
用户后续接受图像偏差并改为辅助仿真用途，approved_for_collection=true仅表示允许近似场景采集，
不表示相机/桌面已校准。每个episode写入scene_fidelity，保留未核实的相机和摆位信息。
本机缺少rs-enumerate-devices及当前venv中的pyrealsense2；本轮不安装工具、不打开相机。

工作顺序（部分准备已完成，批量采集与训练尚未启动）：
1.用户已认可大致布局。
2.相机位姿/内参/裁剪与实际ACT输入仍有偏差；用户允许作为近似场景辅助数据采集，后续可继续校准。
3.专家采集器已适配：按关节名解析控制顺序、使用本场景高度和真实HOME/夹爪动作定义。
4.少量完整抓取/入盒/释放及一条回放通过；按辅助仿真用途，一条完整图像写入检查已通过，300条命令见下。
5.仿真种子及真实源轨迹按组划分训练/验证/测试；仿真失败另存，不充当成功示范。
6.合并已审核的真实专家数据。按域采样，保证真实来源总采样概率高于仿真；比例待数据审计后确定。
    比例待新实物数据审核后确定；避免仿真数量主导采样。
7.加载当前ACT权重做独立微调，保持真实数据基线；检查新旧动作归一化和相机分辨率、关节顺序一致。
8.先做MuJoCo闭环评估及真实录制输入回归，再由用户手动执行实物ACT测试。

本版通过XML/几何/落稳及少量接触抓取检查；尚未证明可采300条成功示范或能改善实物成功率。
MJCF模型引用/相机规范参考：https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html

本轮计划变更：
300条指通过完整抓取、保持、计划释放入盒、HOME及30Hz动力学回放的成功示范数。
最新扰动：桌面前后/左右各±2cm、绕中心水平偏转±5度；方块在桌面坐标中平移各±1cm、朝向±30度。
桌面高度和倾斜不变；盒子/方块随桌面刚体变换，相机及机器人基座固定。
抓取候选朝向随方块变化，具体可行抓取角保存在grasp_world_yaw_rad；不会强行采用不可行角度。
范围为初始工程选择，尚未测得真实误差分布；相机标定偏差仍未解决。
旧实物专家、新摆位实物专家、仿真示范分开保存；每条记录来源和摆位批次。
新摆位应重新示教并确认抓放成功；旧轨迹固定回放不能自动适应物体位置变化。
训练前逐条审核；同一源轨迹的重复采集归同一split，避免训练/测试泄漏。
对比：保留现有模型；用相同起点分别微调“审核后的实物数据”和“同样实物数据+300条辅助仿真”。
两组使用同一实物留出集和任务摆位测试，真实域采样概率高于仿真，比例待数据审核后确定。
仿真成功率仅作仿真指标，不能替代新摆位实物评估。本轮不启动训练或实物采集。

完整图像采集验证：
results/act_fixed_scene_aux100_smoke_20261015：1条55.069秒成功轨迹，独立动力学回放通过。
两路相机各1653帧640x360，右/左动作与观测各1653条；dataset_audit.json记录校验结果。
review.png为初末帧预览。此目录仅为采集验证，不自动并入训练/测试数据。

300条辅助仿真采集（用户手动执行；不需要H1、gateway、ROS或实物相机）：
cd $HOME/openarm_vla_ws
MUJOCO_GL=egl .venv-real-act/bin/python scripts/collect_act_fixed_scene.py \
  --episodes 300 --max-attempts 1800 \
  --table-xy-jitter-m 0.02 --table-yaw-jitter-deg 5 \
  --cube-xy-jitter-m 0.01 --cube-yaw-jitter-deg 30 \
  --output data/sim/openarm_randomized_aux300_20261016

300是成功提交总数，失败不计入；最多1800次尝试，达不到300则报告未完成，不放宽成功判据。
命令未自动启动。Ctrl+C可中止；同一命令、同一代码/配置重跑会保留已提交数据并继续新的尝试。
每次失败保留attempts记录；图像写入中断的未提交临时数据在恢复时清理，不计为成功。
源代码/配置变化会拒绝向原数据目录续写。该输出保持独立，不覆盖历史数据。
采集完成后先审核/划分/转换，再准备两组可对比的训练配置；不自动启动训练。

随机化记录与质量检查：
attempts/<编号>/randomization.json 保存每次具体桌面/方块变换，失败也保留。
metadata.yaml 每条成功示范保存变换、随机种子、抓取世界角；相机仍标记未精确对齐。
coverage.json 按6个区间统计桌面x/y、桌面偏转、方块朝向的尝试/通过/提交数量。
每次按实际旋转盒子坐标判断方块完整入盒并落稳，不使用原盒子世界坐标。
采集完先检查coverage.json；若成功数据集中在部分范围，不能声称全范围已覆盖。
失败不降低判据；未达300会明确报告不足。

2026-10-16 新版验证证据：
results/act_random300_pilots_20261016/summary.json：4种不同随机摆位通过抓放/回HOME与独立30Hz动力学回放；
另2种摆位因无抓取候选或运输IK分支不连续被拒绝，保留失败证据。这不是随机成功率估计。
旧seed_20261018发现三次样条越界；新版在该情况下使用保形插值，v2_seed_20261018复测通过。
results/act_random300_image_smoke_20261016：新版完整图像采集成功；动作/观测及两路相机各1734帧。
results/act_random300_pilots_20261016/randomization_preview.png：4种随机桌面/方块布局预览。
几何变换、旋转盒子成功判据及数据写入测试4项通过；300条批量采集和训练尚未启动。
