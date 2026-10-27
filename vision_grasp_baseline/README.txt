OpenArm V2 独立视觉算法抓放基础测试
更新：2026-10-24（Asia/Shanghai）

当前状态摘要（优先于下方按日期保留的历史记录）
--------------------------------------------
37条closed_transit_v2数据已冻结：nearby30、spread7、rotation0。
训练30条/40772帧，验证7条/9532帧；配置act_closed_v2_37eps_20261023b_50k.json。
用户启动的本轮50K训练已于2026-10-23 20:54:55结束；日志End of training和step=50000均已核对。
checkpoints含005000至050000共10个完整存档，last->050000。
日志最低验证损失：30K=0.1011；50K=0.1015。30K仅为优先离线评估候选，不是已验证实物最优。
当前未有37条模型的分阶段评估/实物抓取成功率报告；旧27条两步smoke报告不能替代。
最新训练结果、证据路径、30K/50K离线评估指令见../docs/ACT_TRAINING_RUNBOOK.md。
H2控制器就绪查询超时尚未彻底解决；旧ACT position-only/张爪起点仍不兼容新示教。
本次只更新文档，未运行训练/推理、未连接或操作机械臂；实际硬件状态未知。

当前采集流程：closed_transit_v2（以下旧日期内容为历史记录）
-------------------------------------------------------
机械零位、空夹爪闭合(-0.02rad；与抓物时的-0.085rad区分)
→ 闭爪到预抓取安全点 → 原地张开(-0.45rad)
→ 张爪垂直接近 → 闭爪抓取 → 搬运 → 入盒张爪释放
→ 保持张爪垂直撤离 → 在安全位置原地闭爪 → 闭爪返回机械零位。
不是到了方块接触位置才张爪，也不是在盒内围着方块重新闭爪。
左右臂零位和右夹爪闭合均验收通过后才开始录制；结束也检查实际反馈和控制器目标。
保留整条轨迹、两路完整640x480图像、16维状态/动作及其时间戳；不剪辑、不做掩码。
四次开合爪均与手臂保持段共用时间线，并继续执行全路径碰撞、速度/加速度/jerk检查。
没有降低既有限位、桌面安全余量或实测碰撞保护。

2026-10-23 新数据快照37条（act_closed_v2_37eps_20261023b）
--------------------------------------------------------
nearby合格30条、spread合格7条，共37条；rotation尚无合格数据。
按完整物理episode及类别划分：训练nearby24+spread6=30条/40772帧；
验证nearby6+spread1=7条/9532帧。失败spread episode_007未开始录制，不参与；
nearby episode_005待审核，不参与。旧27条版本及所有原始记录均保留，不修改其划分。
新旧版本验证划分不同，不能将两版的验证损失直接当成同一测试集比较。
原始记录中同时存在：demonstration_action.jsonl位置指令、target_*.jsonl各控制器输出、
controller_raw.jsonl的output.positions/output.velocities与reference/feedback/error、
joint_observation.jsonl实测关节反馈。runs/cycle_*/continuous/raw_can.jsonl另有CAN指令帧和反馈帧。
控制器输出/本机CAN指令帧记录不等于电机必然按目标运动，仍须与反馈核对。
本版动作监督仍是16维绝对位置目标，速度保留在原始记录，不是额外的ACT输出头。
全部37条累计167788条原生动作标签已逐条核对原始JTC输出、时间因果关系和状态来源；
完整抓放/闭爪回零阶段、两路640x480 RGB、30Hz对齐均保留，无掩码或任务裁剪。
数据：data/lerobot/act_closed_v2_37eps_20261023b；READY.json标记已完成导出。
配置及训练/验证首中末样本读取检查通过；当时的数据整理未启动训练、未查询/操作机械臂。
后续用户已启动并完成本版50K训练；以2026-10-24顶部摘要及最新训练记录为准。

训练此37条版本必须显式选择新配置；train_closed.sh不带配置仍指向旧27条版本：
  cd $HOME/openarm_vla_ws
  OPENARM_CLOSED_CONFIG=configs/training/act_closed_v2_37eps_20261023b_50k.json OPENARM_TRAIN_LOG=results/act_training/act_closed_v2_37eps_20261023b_50k.log ./vision_grasp_baseline/train_closed.sh train
模型输出：outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/
默认50K、每5K保存并全量验证9532帧，不是续训旧27条版本；该运行现已结束，勿原目录重复启动。
该阶段当时只整理数据；此前H2控制器查询异常尚未解决，旧position-only ACT部署兼容问题也仍需验收。

2026-10-23 首版新数据ACT训练（act_closed_v2_20261023a）
-----------------------------------------------------
冻结27条nearby合格数据，22条训练/5条验证；episode_005待审核不参与。
每条物理轨迹与完整规划哈希均去重，按完整episode划分，不以相邻帧拆分验证集。
后续继续采集spread/rotation不会修改本版；另取新name建立下一版快照，不改已有清单。
原始collections及其review不修改；data/lerobot/act_closed_v2_20261023a独立嵌入原始JPEG字节。
保留两路640x480 RGB、完整时间线、原始关节状态/动作因果对齐，不做腕相机掩码或任务片段裁剪。
首轮慢速PNG转换的中间产物已移至该快照下train_partial_png_20261023_174704，未删除、不会用于训练。

输入：顶视RGB + 右腕RGB + 16维实测JointState.position(rad)。
动作监督：未来50帧(30Hz约1.67秒)的16维绝对关节目标，来自JTC output.positions；
不是实测关节姿态，不是MoveIt稀疏路点，也不是末端XYZ。使用前10帧是部署滚动执行约定，非训练只学10帧。
顺序固定：右J1-J7/右夹爪、左J1-J7/左夹爪；实物只控制右8维，左侧保持。
学习完整闭爪出发→预抓取开爪→下降→抓取→搬运→释放→撤离→空爪闭合→回零。
模型不输出ROS协议/碰撞判断/episode完成标志；安全验收、回零停止仍由执行层负责。

独立新入口使用当前本地LeRobot ACT，ResNet18从缓存的ImageNet权重初始化，ACT本身新训练。
默认50K、batch8、chunk50/action_steps10，每5K存档并全量验证；不是从旧20K模型续训。
max_eval_samples必须为0：当前安装版本表示不截断，验证6777帧覆盖全部5条轨迹所有阶段。
拒绝CLI改成256等前缀采样，修正旧训练验证只覆盖第一条验证轨迹开头的缺陷。
训练/验证仅使用训练集归一化统计；不修改site-packages或ACT核心损失。
当前安装版HF列索引会对无关图像触发整行transform；读取50帧action时也重复解码相机。
新入口仅在本地SelectedColumnReader中先选列再选行，原版约274ms/50目标，新版约0.8ms。
目标索引、跨episode的padding、图像/状态与数值不变；已验证逐Tensor相等及spawn双worker。
两种读取方式的前两步训练损失也相同；这是读取效率修复，不是ACT算法准确率修复。
首版只有nearby，不是大范围位移/旋转泛化的证据，50K也不保证实物成功。

准备（已导出后仅核对冻结来源；不合并后来新增数据）：
  ./vision_grasp_baseline/train_closed.sh prepare
短测试（2步训练、全量验证、保存独立smoke输出，不作为可部署模型）：
  ./vision_grasp_baseline/train_closed.sh smoke
正式训练（用户手工启动，不自动占用数小时GPU）：
  ./vision_grasp_baseline/train_closed.sh train
配置：configs/training/act_closed_v2_20261023a_50k.json
训练日志：results/act_training/act_closed_v2_20261023a_50k.log
模型：outputs/act_closed_v2_20261023a_50k/checkpoints/<步数>/pretrained_model
最近存档：outputs/act_closed_v2_20261023a_50k/checkpoints/last/pretrained_model
断点恢复（从最后一次保存的5K倍数恢复，不是从中断瞬间恢复）：
  OPENARM_CLOSED_CONFIG=outputs/act_closed_v2_20261023a_50k/checkpoints/last/pretrained_model/train_config.json ./vision_grasp_baseline/train_closed.sh train --resume=true
全验证集分阶段离线预测(rad，含“保持当前姿态”基线；不是实物成功率)：
  ./vision_grasp_baseline/train_closed.sh evaluate --checkpoint outputs/act_closed_v2_20261023a_50k/checkpoints/last/pretrained_model --output results/act_training/act_closed_v2_20261023a_last_phase_eval.json
比较5K/10K/后续存档应看完整验证及闭爪/释放/回零分阶段误差，不默认步数越大越好。

训练入口无ROS/CAN/相机/动作连接；nice10、idle磁盘优先级、OMP/BLAS线程1、训练worker2。
这些限制降低资源争抢，但不保证同机实物采集无调度延迟；建议先确认空载/试采稳定，
若并行时再现CAN或碰撞响应超时，先停止训练/采集定位，不放宽安全门槛。
重要：新示教右臂为position+velocity接口；模型学q序列，部署需生成并验证一致的q/qd与时序。
旧ACT position-only执行器及旧张爪HOME配置不兼容；本次只建立训练入口，不能直接actctl start。
训练后先做存档分阶段离线评估，再完成新闭爪启动/控制器接口匹配的实物部署验收。
本次验证：采集/安全回归468项通过、2项因环境缺少训练依赖跳过；训练环境31项通过。
实际GPU短测试outputs/act_closed_smoke_20261023_175210完成2步训练、6777帧全量验证、
checkpoint保存和重载；分阶段离线推理也完整处理6777帧，报告位于
results/act_training/act_closed_smoke_20261023_175210_phase_eval.json。
smoke仅验证流程，2步权重没有训练充分、绝不能用于实物；旧27条50K输出目录截至2026-10-24未存在。
不要把本段旧27条未训练状态与上方已完成的37条50K混淆。

2026-10-23 闭爪接近净空与执行线程阻塞修复（cycle_021/022）
--------------------------------------------------------
021在reach阶段、预抓取开爪之前检测到ee_link1/方块模型碰撞；
该轮最大实测末端/指令末端差约21.6mm，不能将这次碰撞当作正常抓物而放行。
原先闭爪接近仅要求不相交，现规划时将40mm方块各方向外扩30mm，
预抓取开爪扫掠和最终平滑后的完整控制器插值均检查此净空；
到张爪下降approach开始才恢复原尺寸，不改真实方块位置、抓取标签或接触允许规则。
新轨迹带pickup_clearance_m=0.03；旧缓存没有此字段时拒绝，需重新完整规划。
净空仅是规划余量，不是对实际跟踪误差/连续碰撞安全的保证；实测碰撞仍停止。

022前三次实测碰撞响应均很快，随后reach阶段响应年龄318ms超过200ms；
旧日志不足以唯一判定是服务处理慢、主线程阻塞还是系统调度停顿。
已确认原执行循环同步等待记录器status/event HTTP，最长可等2秒且期间不spin ROS，
现在移到有界后台I/O线程，主线程非阻塞检查错误；不改200ms碰撞超时或100ms CAN门槛。
记录器健康仍每250ms查询，按请求发出时间计龄，超过750ms或HTTP/写盘错误即失败；
事件队列32项，溢出拒绝，成功必须排空事件；事件detail保存source_monotonic_s。
后台线程不发任何机器人指令。阶段切换不能吞掉已收到的碰撞，取消请求同时清理客户端pending。
result.json增加碰撞请求关节状态、阶段、轮询间隙、future是否完成，帮助区分下次故障。
场景切换仍含同步MoveIt服务调用；此修改不能保证所有调度/服务超时永不再发生。

验证：452项离线回归通过，包含320ms记录器阻塞时碰撞轮询继续、真实迟到仍停止、
阶段切换保留碰撞、完整插值检查外扩方块、碰撞失败后还原名义场景等。
未发送实物动作、未清故障、未重启H1/网关。Python子进程下次执行即加载，不需要重新安装硬件库。
现有合格nearby共18条，可继续追加，不reset-collection、不用--replay-plan复用旧路径。
恢复前由操作者检查实际停止姿态、接触/受力和网关状态；先试采1条，合格后再补11条到30。
若fault且drain_ready=true才考虑手工reset-fault；仍有未解决目标/碰撞/状态异常时不要强行复位。

首次启用（由操作者执行，本次未自动安装、未启动硬件）：
  cd $HOME/openarm_vla_ws
  # 机械臂断电并支撑好；停止旧H1和gateway后再安装。
  ./vision_grasp_baseline/run.sh install-h1-fix
安装器校验候选库/源码/测试摘要，拒绝控制进程仍在运行时替换；原库备份到results。
候选库：.closed_transit_h1_candidate_20261023/install/openarm_hardware/lib/libopenarm_hardware.so
断电安装阶段可执行 ./scripts/actctl prepare-host 修复启动所需CAN锁目录。
随后由操作者按已验证参数配置can0/can1并上电，确认夹爪内无物、机械臂处于安全零位、
工作空间无人且急停可用；H1激活会使能电机并执行原有机械回零过程。
  终端1：./vision_grasp_baseline/run.sh h1
  终端2：./vision_grasp_baseline/run.sh gateway
  终端3：./vision_grasp_baseline/run.sh status
         ./vision_grasp_baseline/run.sh collect --episodes 1 --tag nearby
每个终端先cd到本工作空间。等待控制器全部激活后再采集；collect会在网关空闲且无故障时
同步当前保持目标。首轮只采一条，观察全过程，成功后人工输入y验收，再扩展采集。
验收输入：y=合格并继续；n=保留待审核、不计合格、继续补采；q=保留待审核并结束本次采集。
--episodes表示本次新增合格条数，正常执行但人工拒绝的轮次不会消耗合格配额。
输入n后回到摆放提示，必须再次回车才开始；无效/空验收输入会重新询问，不自动确认或运动。
技术故障仍立即停止并生成失败标记，不自动恢复、不跳过安全检查。
本次未更改CAN波特率、摄像头序列号、机械臂增益、重力补偿或夹爪接触模式。
停止采集：./vision_grasp_baseline/run.sh stop（不是硬件急停，不关闭ROS）。
结束H1之前必须先支撑双臂；正常ROS退出会停用电机。

重要区别：./scripts/actctl h1仍是旧ACT模型的张爪启动入口，不用于本次新采集。
recovery-h1仍保持测得的初始位置，不代表已闭爪；collect的zero-test会在零位闭爪并验收。
所有新完整轨迹带gripper_policy=closed_transit_v2；包括v1在内的旧缓存或--staged录制直接拒绝。
默认每轮新规划，不能复用以前的张爪出发轨迹。
保存路径：collections/<nearby|spread|rotation>/episodes/episode_XXXX；
执行记录：同类别runs/cycle_XXXX；会话日志：同类别sessions/session_<类别>_<日期时间>。
2026-10-23重新开批后使用三位编号episode_000/cycle_000，各类别独立从零开始；
旧试采数据与40轮旧执行记录已移到collection_archives/reset_20261023_141909_780774，未删除。
正常采集会继续追加，不能每次都清零；失败/中断也保留编号，避免覆盖证据。
失败可见标记位于同类别failed/episode_XXX_FAILED/FAILED.json，中断为INTERRUPTED。
标记记录原因及原始数据/执行/会话路径；原始文件不自动搬动，避免中断子进程仍在写入。
失败metadata强制pending/diagnostic_only，不进入合格训练数据；未确认成功则REVIEW_REQUIRED。
需要另开一批时：run.sh reset-collection --tag all先预览，追加--archive-existing才整体归档；
归档拒绝正在运行的采集/cycle，不发送硬件命令。本次存储与采集回归246项离线测试通过。
不传--tag默认nearby。编号避开已有episode及cycle，不覆盖/删除旧数据或日志。
相机检查默认放到runs/camera_checks。显式--output仍是全新的独立会话目录。
新数据导出可显式指定 --gripper-policy closed_transit_v2；导出器拒绝混合新旧夹爪协议。
旧train.sh prepare/train仍绑定20261022冻结的32条数据/旧配置，不会自动训练新session。
新采集验收后需为新数据单独创建训练划分、训练配置和闭爪启动部署配置，不能直接混用旧模型。
闭爪改变的是示教中的视觉分布，不能保证消除腕相机干扰或保证ACT成功；仍需实物评估。
最新验证：356项Python测试通过、1项跳过（分采集/训练两个虚拟环境运行）；
此前28项C++测试通过，硬件库本次未改动。
五段轨迹实际TOPPRA离线定时通过。
无实体抓放测试，首轮仍需操作者验收。下文的成功记录、数据数量和旧指令均为历史情况。

2026-10-23 控制器查询恢复修复（cycle_014）
----------------------------------------
本轮完整抓放在控制器就绪查询阶段失败，尚未开始录制或下发完整轨迹。
网关原先只等list_future.done()，没有请求超时；一次永不返回的响应可阻止后续全部查询。
现只读查询750ms超时后移除pending并取消future，间隔250ms重试；不并发堆积请求。
已完成但迟到的响应同样丢弃，不能重新刷新缓存；状态年龄保守地从请求发送时刻计算。
status增加controller_query：fresh、age_s、pending_age_s、service_ready、timeouts、last_error等。
控制器列表原有1秒新鲜度门槛保留，缓存active过期后不能获取控制权或接受新的H2轨迹。
轨迹执行前仍要求真实active响应：单次查询3秒、最多3次、整体最多12秒的软件等待预算。
失败重建的仅是ListControllers只读客户端，不重启控制器，不操作动作客户端或自动重放轨迹。
实际inactive直接拒绝，不作为可忽略的网络失败；持续服务失联也拒绝，不放宽CAN/碰撞保护。
continuous/result.json记录controller_readiness诊断，便于区分服务发现、响应超时与inactive。
这一修复处理卡死后的恢复，不能据此断言首次DDS/服务异常根因已被消除。

Python修改不需要重装硬件库；网关必须由操作者正常退出后重启才能加载新代码。
保持双臂安全支撑，不在有运动/未解决目标时强杀进程。H1正常运行则无需为此修复重启H1；
H1已关闭时，只有确认空爪及安全零位后才使用本目录h1入口重新启动。
终端1（仅H1已停时）：./vision_grasp_baseline/run.sh h1
终端2（旧网关已退出）：./vision_grasp_baseline/run.sh gateway
终端3：先status等controller_query.fresh=true且三个受控控制器active，再synchronize-hold。
再次status确认idle、fault=null、synchronized=true且pending/hold_pending/raw_active均空。
先collect --episodes 1 --tag nearby验证；本次修复前合格12条，若补采1条合格后还需17条到30。
不要再次reset-collection，不覆盖已保存数据；技术故障仍停止，不能反复清故障强行继续。
验证：436项离线测试通过；ROS域194假控制器集成测试13项检查通过，不连接CAN。
未对本次查询修复做实物复测，也未重启或清除物理网关故障。

2026-10-23 CAN接收进程隔离（cycle_001再次延迟后的修正）
----------------------------------------------------
cycle_001的zero-test再次失败：4928帧、约1.032秒记录中出现约231ms读取间隙，
12帧积压超过100ms，最大约232ms；wall/monotonic偏移变化小于1微秒。
这说明之前仅把日志移至线程还不足够，不能宣称问题已彻底修好。
现将采集的CAN观察放到独立轻量Python子进程，避免共享ROS/规划进程的GIL；
具体停顿由GIL还是操作系统调度导致，原始日志本身不足以唯一判定。
zero/grasp/continuous_execute/waypoint_place/place采集入口均使用isolated_telemetry。
子进程只加载原有只收不发的CAN观测器，不初始化ROS、不获取控制权。
父子进程仅以AF_UNIX请求新快照，不回放旧缓存；父进程再次核对原始源时间戳。
历史解码器间接导入NumPy，但子进程的BLAS/OMP/MKL/NUMEXPR线程上限设为1，避免额外线程池。
100ms内核源时间戳保护不放宽；IPC超时、子进程退出、原始CAN错误均拒绝。
子进程内部仍保留有界日志队列；它不能保证Linux调度永不停顿，也不是硬实时保护。
每轮raw_can文件旁增加.observer.log，仅用于子进程异常诊断。
不需要重装硬件库或为了此Python修改重启H1/网关；下一次子进程会加载新代码。
本次未启动真实CAN观测子进程、未发送动作，实体效果仍需操作者复测。
独立进程测试使用合成反馈：父进程持GIL阻塞250ms时，观察子进程继续更新；
另覆盖真实过期/未来反馈、过期指令、IPC超时及迟到响应、子进程崩溃和退出清理。
本次采集与CAN回归258项离线测试通过，未做实物复测。

精简说明：被动CAN日志不发送总线帧，删旧文件不能减少CAN通信；日志降采样主要降低
CPU/磁盘压力。当前采集入口已关闭RViz，不需要同时运行旧ACT策略/Dora或额外实时查看器。
不能为了减负删掉反馈新鲜度、碰撞/停止保护、动作标签、关节状态或双相机同步数据。

2026-10-23 CAN接收与ACT标签核查（上一轮线程修复）
-------------------------------------
cycle_0017在zero-test开始录制前失败，不是物体摆放相似或轨迹重复导致。
原始日志显示一次约247ms的用户态CAN接收停顿；内核时间戳与读取时间相差约247ms，
wall/monotonic时钟偏移没有对应跳变。不能仅凭这些证据认定电机掉线或磁盘是唯一根因。
原接收线程同步做JSON编码、写文件及flush，存在阻塞接收的风险。
现将这些操作移至独立日志线程，接收线程仅非阻塞入队并维护新鲜状态缓存。
队列有上限，溢出/磁盘错误仍锁存失败；正常关闭会排空日志，不静默丢帧。
100ms内核源时间戳、反馈失鲜、CAN错误、收包溢出及碰撞保护均未放宽。
此改动消除了接收线程中的磁盘I/O，但不保证普通Linux/Python调度永不出现停顿；
再次出现真实过期反馈仍必须停止，不能自动重试或自动清故障。
Python修改不需要重装硬件库，下一次zero-test/continuous-execute子进程自动使用。

本采集控制契约：
  observation.state = /joint_states的实测位置，16维，rad。
  action = 三个JTC的output.positions按源时间因果对齐，加被动左夹爪初始保持值。
  不是把实测姿态当动作，也不是直接用稀疏规划路径作为实际动作标签。
  右臂位置+速度前馈，右夹爪/左臂位置；原始速度仍保存在controller_raw.jsonl。
  LeRobot ACT取当前观测及t,t+1/30,...的动作块，未做“改为反馈标签”的处理。
  目标与实测有偏差是跟踪诊断项，不能因此给所有标签减固定偏置或改成实测值。
  旧ACT部署只接受位置接口；新模型必须配匹配的控制/时间/滤波契约并验证后再上机。
  本次没有放开旧部署入口，也没有声称位置+速度ACT实物部署已验证。

只读核验（不连接ROS/CAN、不改原始数据）：
  ./vision_grasp_baseline/run.sh audit-act vision_grasp_baseline/collections/nearby/episodes/episode_000
逐条比对原始ROS关节反馈、原始JTC输出、各控制器目标流、最终动作标签和源时间；
缺证据、换成反馈标签、未来时间对齐或控制接口不符时拒绝。
closed_transit_v2的export_act已强制执行该核验，并在转换报告记录控制契约与跟踪统计。
上一批episode_0015已实际离线核验：4431条动作、4432条状态通过，标签确为控制目标；现已归档。
这只证明该样本的记录/转换契约，不证明旧70条数据或模型实物成功率。

2026-10-23 规划状态读取修复：cycle_0010在抓取下发之前因GraspNode.full_joints不存在失败。
完整关节状态读取现放到共用GraspNode，复用原/joint_states订阅；七轴手臂和夹爪必须来自
同一条有效消息，刷新状态时不接受旧缓存。执行节点不再用另一订阅覆盖这份已验证状态。
新增真实构造/订阅绑定、缺夹爪、非有限值、过期缓存及plan_geometry入口测试。
这是Python子进程修复，无需重装硬件库或重启H1/网关。先检查status为空闲、无故障且已同步，
然后重新运行collect --episodes 1 --tag nearby；保留cycle_0010证据，自动续用新编号。

2026-10-23 空爪回程闭合修复：原v1复用了抓物目标-0.085rad，且允许0.04rad偏差；
回看出现目标约-0.087、实际约-0.120rad的明显开口仍能回程。
v2单独设置空爪目标-0.02rad、反馈范围[-0.055,0]rad；不顶0rad硬端点、不增加增益。
cycle_0013目标-0.02rad、实测稳定-0.0494rad，操作者已确认肉眼闭合；因此仅将
原反馈下限-0.04调整为-0.055rad（保留约0.0056rad测量余量），上限仍为0。
目标、电机增益、稳定时间、碰撞检查均未更改；-0.085/-0.1204rad仍不能通过空爪验收。
metadata记录实际采用的empty_transit_feedback_range_rad，避免将误差容限误读为对称放宽。
抓物及搬运仍用-0.085rad，放置张开仍用-0.45rad。
安全撤离后，在闭爪运动结束的静止目标上保持1秒；回程前0.25秒检查连续至少0.30秒
的新鲜闭合反馈，未满足则请求同时取消两个轨迹；回程中张开/反馈失鲜同样中止。
这仍是现有同步时间线的软件监控，不是硬实时安全互锁或连续碰撞认证。
开始录制前zero-test同样关闭到-0.02rad并验收稳定反馈；最后回零规划显式带入该夹爪姿态。
H1硬件启动保持仍为-0.085rad，本次不重装、不重启硬件；收紧动作在zero-test录制前完成。
旧v1原始数据和审核标记不自动修改，不与v2混为同一训练数据版本。
本次无实物操作；新目标能否达到、图像中是否充分闭合，仍需先采1条人工验收。

2026-10-23 实时碰撞服务启动修复：cycle_0014在第一条实时RPC上超时，checked_states=0。
执行节点采用已有全路径核验证明时，之前没有对自己的/check_state_validity客户端做服务发现/预热。
现改为录制及下发动作之前等待服务发现（整体最多10秒），并要求连续两条有效响应在200ms内完成。
冷启动慢响应只在尚未运动时允许用于预热，不作为运行时合格响应；持续不可用则在下发前失败。
运行中仍为200ms超时，不关闭碰撞检查、不放宽几何或关节限位；即使RPC等待中也检查新鲜CAN与硬限位。
请求计时改为实际发送时刻，场景接触规则更新后的状态重新读取，避免把发送前耗时误算成RPC延迟。
result.json记录startup_prepared/startup_probe_count/startup_elapsed_s/runtime_timeouts等诊断字段。
仅Python子进程修改，不需要重装硬件库。187项离线测试覆盖冷发现、慢首次响应、持续失联、
碰撞拒绝、真正运行超时以及启动失败不录制/不下发动作；尚未对这次修改做实体复测。

旧32条数据的训练入口：docs/ACT_VISUAL32_TRAINING_20261022.txt
collections现按nearby/spread/rotation整理，共32条已审核合格实物轨迹。
./vision_grasp_baseline/train.sh prepare：固定划分26训练/6验证并导出。
./vision_grasp_baseline/train.sh smoke：2步真实数据GPU训练/验证检查。
./vision_grasp_baseline/train.sh train：独立ACT初版20k训练，不混入旧数据。
详细路径、日志、检查点和恢复方法见上述文档；下文旧会话路径仅作历史记录。

测试数据清理：按用户要求，collections中此前6组测试采集（含成功测试）已删除，
释放1,566,795,004字节。下文旧collections路径仅作历史记录，文件已不再存在。
数值验证摘要及清理清单保存在runs/pose_fix_evidence_20261021；回归测试所需的
4份小型场景JSON已独立放在tests/fixtures，测试不再依赖待清理的采集目录。

目标和范围
----------
从机械零位开始，视觉识别蓝色4cm方块和黑色开口盒，统一抓取/释放规则，
现已接入重复连续采集、零位返回与LeRobot转换。旧分段流程已完成一轮实体成功；
新完整轨迹流程已于20:20完成一轮实机抓放回零和双相机采集验证。
本目录独立于旧ACT、模型权重、历史训练数据和openarm_mtc_ws。

新采集入口（2026-10-21晚间软件更新）：
  ./vision_grasp_baseline/run.sh camera-check --seconds 15   # 无运动，双相机落盘检查
  ./vision_grasp_baseline/run.sh collect --episodes 1       # 已启动H1/网关后先测1轮
  ./vision_grasp_baseline/run.sh collect --episodes 30      # 每轮人工复位方块
  ./vision_grasp_baseline/run.sh collect --episodes 10 --tag nearby    # 常用位置附近小幅平移
  ./vision_grasp_baseline/run.sh collect --episodes 15 --tag spread    # 已验证范围内前后左右分散
  ./vision_grasp_baseline/run.sh collect --episodes 5 --tag rotation   # 已验证范围内小角度旋转
历史保存方式（已于2026-10-23改为上述分类追加）：--tag 曾将默认目录命名为 collections/session_<tag>_<日期时间>/，并写入 session.json
及每轮 episodes/episode_XXXX/metadata.json 的 collection_tag / collection_description。
标记仅用于数据分类，不改变运动、验收规则或允许抓取范围；不传 --tag 时目录命名不变。
显式传 --output 时使用指定目录名，类型仍写入元数据。已有数据不自动改名。
2026-10-22桌沿跟踪余量：新完整规划在固定桌面包围盒四周额外保留30mm，
平滑后的全轨迹也按该余量核验；实机实际反馈碰撞监控仍使用原固定包围盒。
trajectory.json记录table_edge_clearance_m。旧缓存不具备这项余量，应重新规划。
此修改通过71项离线测试，尚待实机复测；不能据此认定实际接触或定位误差已消除。

2026-10-22位置/速度指令修复（需重新启动H1和网关）：
recovery-h1默认使用关节4启动下限-0.08rad，无需额外参数；任务规划仍保留-0.02rad下限。
  终端1：./vision_grasp_baseline/run.sh recovery-h1
  终端2：./vision_grasp_baseline/run.sh gateway
  终端3：./vision_grasp_baseline/run.sh synchronize-hold
         ./vision_grasp_baseline/run.sh status
  等待idle、fault=null、synchronized=true、drain_ready=true；确认
  controller_command_interfaces.right_joint_trajectory_controller为[position, velocity]。
  然后：./vision_grasp_baseline/run.sh collect --episodes 1 --tag nearby
右臂JTC通过局部参数文件同时下发位置与速度，末端夹爪及左臂保持原接口。
网关保持目标显式发送零速度/零加速度。采集元数据记录实际控制接口。
现有逐步过渡的几何规划不变；没有改为高位折线路径，没有调高电机增益。
重力补偿保持启用；末端质量/重心未经实测核对，本次没有猜测负载或增加补偿倍数。
回退：关闭本次H1/网关后，终端1的recovery-h1命令加 --position-only，
随后重启网关并重新synchronize-hold。不要运行中切换接口。
停止保持完成阈值为0.02rad/s，以覆盖实测静止反馈约0.01099rad/s残值；
同步进行中的manual_hold_synchronization状态等待完成，超时仍拒绝开始采集。

2026-10-22放置下降LIN生成失败修复：真实MoveIt日志显示右J5加速度
0.580296rad/s²超过0.5rad/s²，Pilz仅返回99999。相同记录场景已在隔离
GenericSystem环境复现。LIN几何生成失败时，执行前最多重试4次，依次使用
原生成速度/加速度比例的1、1/2、1/4、1/8；起点、目标、姿态及限位保持一致。
成功后仍进行原直线/姿态/碰撞核验，并用TOPPRA重新计算整轮执行时间，
执行中不增加规划或停顿。geometry.json的generation_attempts保存尝试记录。

2026-10-22执行前自动恢复：
  collect命令不变。每个识别阶段先采10帧，只有时间一致性不合格时才自动
  再采最多2批20帧；所有批次保留，失败批次保存scene_unstable.json和标注图。
  最新帧一致性、位置/角度与桌面范围门限仍执行，持续不稳定则停止。
  完整预规划失败时，最多重新生成3个完整候选；第一个通过限位、几何、平滑
  和碰撞检查的候选进入执行。目录complete_plan、complete_plan_retry_1/2。
  cycle_result.json记录perception_attempts和planning_attempts。
  重采、候选规划均在录制/抓取之前；实际运动中失败不会自动继续抓取。

按操作者确认的静止场景采集（collect默认，无需增加参数）：
  回零后识别 -> 完整预规划（最多3个候选）-> 全路径核验 -> 记录并连续执行。
  新规划沿用回零后的同一组场景，省去规划后的execution_scene重复采图。
  静止物体使用整批至少70%一致帧的中位估计，不再要求最后3帧中2帧一致。
  scene.json记录pose_estimation.stationary_fixture；该策略在后续核验中保留。
  旧缓存回放仍重新采图核验。直接cycle可用--stationary-fixture启用相同策略。

collect首次启动会在网关idle、无故障、无待处理/活动轨迹、尚未同步时，
自动发起一次synchronize-hold并等待完成；无需每次重启后手动补这一条命令。
这是操作者启动collect后的当前控制器位置保持同步，不会发起HOME/回零。
运行中的阶段交接不会自动重同步，也不会自动清除网关故障。

夹爪开口设置：gripper_settings.py统一配置。2026-10-22按操作者要求，
新连续采集张开目标从-0.35改为-0.45rad，回零前张开/规划/播放保持一致；
闭合目标仍为-0.085rad。实际执行前仍检查完整夹爪运动的碰撞及动态限位。
这项张开幅度修改通过86项离线测试，尚待实机验证，先采1条spread确认。

2026-10-22接近阶段推走方块修复：cycle_0005录像显示方块在接近约6秒时
被手指推走，原实现对整个reach阶段开放了手指/方块接触许可。
新规划和完整轨迹核验在close事件前禁用该许可，运行时测量检查也一致；
只在close开始后允许抓取接触，carry阶段按原规则处理附着物及支撑面。
旧失败轨迹现于5.902秒被手指/方块碰撞检查拒绝；同场景新轨迹通过3269
状态检查，预计34.95秒。89项测试通过，尚待修改后的实机复测。

2026-10-22识别估计重复核验修复：perception原先把measured直接引用到
frames[-1]，估计中位值时改写了原始最后一帧，可能导致首次7/10通过、保存后
复核变成6/10失败。现在两处都显式分离估计结果与原始帧；真实失败图像已通过
修复后的重算及连续10次一致性复核。规划输入不稳定单独返回退出码3，不再
按随机规划失败重试相同输入。历史失败数据与原始图像保持原样。

2026-10-22执行前控制器查询超时恢复：连续执行器用一次ListControllers同时
检查右臂及右夹爪，未收到响应时最多查询3次，每次等待3秒，丢弃旧请求后再试。
收到inactive/missing则立即拒绝，持续无响应也拒绝；通过后才开始记录/发送目标。
93项测试通过；当前实机服务连续5次只读查询通过，未执行机械臂运动。

2026-10-22搬运高度/旋转中心修复：部分分散位置的固定姿态额外抬高3cm无IK，
降速不能解决。现依次尝试额外3cm、1.5cm、0cm，最低候选不低于抬升及释放
TCP高度；释放目标本身仍使方块底部高于盒沿20mm。失败候选的部分段不拼入
成功路径，各候选均保留直线/姿态/碰撞/限位核验，最终全路径再核验。
换姿态时Pilz LIN约束抓取TCP，而非偏置的手腕EE，防止抓取点扫出圆弧。
geometry.json记录transfer_extra_clearance_m及rejected_transfer_heights。
95项测试通过；修改需新的collect规划生效，尚待实机验证。

2026-10-22旋转方块候选修复：预抓取只尝试cube_yaw+90°一个朝向，部分方块
旋转角会使该朝向无IK。现按方块四重对称尝试0、+90、-90、180°等价夹持方向，
必要时再尝试预抓取高度0.11、0.09、0.075m（相对桌面）。各候选的预抓取及
下降必须通过原碰撞和关节检查；闭爪前仍禁止手指穿过方块。所选候选记录于
geometry.json第一段grasp_candidate，包括失败原因、方向及高度。
真实-40°失败场景已在隔离环境复现，换到等价方向后整轮检查通过。
96项测试通过，尚待修改后的实机抓取验证。
  ./vision_grasp_baseline/run.sh collect --episodes 30 --replay-plan /绝对路径/trajectory.json
默认完整预规划 -> 保存 -> 连续播放 -> 回零保存；--staged才使用旧分段流程。
20:04故障修复：规划/执行共用包含10帧波动的固定保守桌面包围盒；continuous-check
在申请运动权限前检查新场景。旧缓存需重新生成，新执行仍不中途规划或停顿。
20:15角度误报修复：方块/盒子使用10帧一致性中位估计，不再只取最后一帧。
要求至少70%帧和最近3帧中的2帧一致；真实位置/角度变化仍拒绝缓存回放。
新完整轨迹执行时不再调用规划器；记录从全部检查结束后开始，末尾视觉验收移到记录外。
机械臂/夹爪各一个同步目标，保留必要开合停留，取消共线中间点逐段停车。
保持零位时无运动联合检查：
  ./vision_grasp_baseline/run.sh recording-check --seconds 60 --plans 5
已完成60秒/5次预规划及20秒/3次预规划+同服务再次记录检查。
完整启动、采集、导出指令：docs/VISUAL_ACT_COLLECTION_20261021.txt
频率：控制100Hz配置，实测状态目标100Hz，JTC动作按原始发布（17:40实测约100Hz），相机/ACT数据30Hz。
新数据为双640x480，不与旧640x360检查点直接混用。

技术记录：docs/VISION_GRASP_BASELINE_TECHNICAL_20261021.txt
本轮证据汇总：waypoint_physical_test_20261021.json
参数/状态清单：config/baseline.json（说明清单，不是所有代码参数的动态配置入口）

最新验证结果
------------
20:32回零中断修复（本次用户关闭实机，仅离线/仿真验证）：
真正触发项是TCP跟踪误差>25mm持续0.2秒；本轮约29mm，随后主动cancel撤销租约，
引出no live authority续租错误。不是关节限位或电机速度超限。
默认连续执行取消该固定跟踪距离停机阈值，保留误差记录；新增实际CAN反馈姿态的
URDF硬位置限位和MoveIt碰撞检查（目标50Hz），反馈失联、故障、STOP仍会停止。
抓取TCP按config/grasp_target_offset.json向world +X（机器人前方）补偿2mm；
只改变抓取目标，物体/桌面/盒子模型不平移，完整附着和放置变换随新抓取重新计算。
轨迹平滑导致碰撞时，执行前自动减小平滑范围并重验，不修改运行中的轨迹。
修复软限位投影后硬改端点引起的人工jerk尖峰；同一几何路径88.9068s降至37.3204s，
保持原配置的速度/加速度/jerk界限。未删除已安装MoveIt的加速度0.5rad/s²、jerk5rad/s³，
也未修改共享URDF、控制器或电机参数；这些不是本轮故障来源。
70项测试通过；原失败记录1586个实际反馈姿态全部通过限位/碰撞复核。
runs/return_guard_mock_20261021_01/forward_2mm_plan_03：新轨迹33.7533s，3181状态预检通过；
mock_execution_03.json：隔离域163、仅GenericSystem完成整轮，1597次实际模拟状态检查，
机械臂/夹爪各一个目标、执行中规划0次，最终回零。新2mm补偿尚未实机复测。
异常退出先保存原始原因，租约监督器给子进程清理时间，不再用续租异常覆盖原始原因。

最新验收标准（用户确认）：方块放进盒内即可，不要求居中或额外3mm盒壁余量。
verify_placed采用inside_bin_v2，仍检查内孔包含、高度、多帧证据和采集质量。
下述shifted_cube测试已用原始10帧RGB-D复核通过并按用户确认标为approved；
acceptance_review/result.json保存重新验收依据，原执行错误报告和原始数据不改写。
其episode已通过ACT导出前审核；新采集默认每轮识别并规划，可在首轮位置附近
4cm范围内调整方块，盒子保持不动。这是允许偏移门槛，不代表该整个区域已实测。
当前实体证据为原摆放和约9.5mm/12.6度变化后的两种摆放，尚未完成30轮成功率统计。
runs/shifted_cube_collection_20261021_01：微调位置后的实机检查，方块平移约9.5mm、
旋转约12.6度，已重新识别和规划。实际抓起、入盒释放、回零完成，轨迹33.1913秒，
记录34.2539秒，数据质量通过。抓取/释放时TCP位置误差约3.10/4.50mm。
当时执行报告success=false：末尾方块靠近盒壁，名义内孔剩余最小间隙1.26mm，
未满足旧3mm验收余量；现已按上方新标准及用户确认重新验收为合格。
照片显示方块已在盒内，这不能代替其严格放置余量验收。网关已恢复idle，无fault。
以下旧collections记录已按上方说明清理：
collections/agent_pose_fix_20261021_01：整轮连续实机抓放回零成功，自动验收通过；
轨迹32.8862秒，记录34.0354秒，机械臂/夹爪各一个目标，执行中规划调用0次。
状态/动作约100Hz，全局29.979Hz、腕部29.923Hz，所有数据质量检查通过。
查看搬运及结束照片确认方块被抓起并释放在盒内；10/10帧入盒验证通过。
数据保留pending，未替用户标记operator_marked_success；session.success=false
表示未凑齐用户审核通过的episode数量，cycle_result.json的success=true。
本轮结束时网关idle且无fault；方块现位于盒内，下次需先放回盒外原位置。
collections/session_20261021_180151_587587：完整实机抓放回零成功，用户确认；
记录224.72秒，状态/动作100Hz、相机30Hz，质量通过。
runs/continuous_mock_20261021_01/plan_07：新轨迹32.648秒，3027状态插值碰撞检查通过，
隔离域162的GenericSystem控制器完成整轮回放；机械臂/夹爪各一个目标。
此候选时间不能当作新实机抓放成绩。实机首轮使用上方collect --episodes 1。

以下为16:44历史验证记录（保留故障与续接证据）：
本轮目录：runs/cycle_20261021_164451。
抓取及抬升成功；两处放置程序错误修复后，place_retry_02完成13段中间点放置，
10/10帧RGB-D验证方块在盒内。13段计划运动时间合计约126.76秒，不含规划等待、
夹爪和相机；最大实时TCP位置跟踪差约14.47mm，最大分段终点位置差约9.85mm。
随后另行调用zero-test完成机械零位返回：202个轨迹点、403个模型状态检查，
运动约20.04秒，右臂最大实际零位偏差0.01469rad（约0.84°）。
该回零记录末尾16个电机反馈均为status=1，左臂最大绝对位置约0.01545rad。
这些是最后一次有证据的状态快照，不代表此刻硬件仍在运行。

用户仍报告动作不够流畅、首次接近抓取位置时接近桌面。
本轮不是合格的批量专家轨迹，也没有验证一条命令从头到尾不中断地完成。
原始cycle_result.json仍为false；续接放置和单独回零成功另有各自记录，原文件保留。

16:44当时已实现与待完成（历史状态；当前完整轨迹实现见上方）
--------------
已实现：视觉位置检测；机械零位起步；抓取/抬升验收；13段搬运与松爪；
实测位置保持启动；规划回机械零位；真实TCP到位、电机反馈和停止请求检查。

待完成：
  1. 轨迹衔接平顺：当前各段独立规划和执行，段端速度为零，段间还有规划等待。
     这是停顿的代码依据，尚不能把所有实物抖动或不流畅都归因于这一项。
  2. 新cycle已在放置成功后自动回机械零位，仍待完整实体复测；单独waypoint-place-test不回零。
  3. 用最终每侧3cm桌面边缘余量复测完整接近/抓取/放置路径。
  4. 新连续双相机/动作/状态采集已完成软件验证和相机实测，待不中断整轮实体测试。

抓取位置是否固定
----------------
位置不是固定坐标。每轮cycle先采集全局RGB-D，检测方块/盒子；回零后再次识别。
抓取XY来自检测结果经原标定和本目录局部修正后的世界坐标。
固定的是竖直抓取和规范化yaw规则、相对桌面的TCP高度、夹爪开闭目标。
放置XY来自检测盒子；TCP释放位置还按实际夹持变换和方块包络重新计算。
腕部相机本轮用于人工/程序读图核验，尚未参与当前物体三维定位算法。

桌面模型：最终每侧3cm
--------------------
默认在grasp.py的precise_scene中定义：table_edge_padding_m=0.03。
XY尺寸=max(检测尺寸, [0.4,0.6]m)+[0.06,0.06]m；桌面高度和原3mm顶面余量不变。
因此3cm是相对最小尺寸基准的每侧膨胀，不是重新测量出来的桌面边界误差。
本轮单独实体回零开始时用的是每侧5cm；执行中用户改为3cm。
回零完成后已无运动地刷新场景为3cm：table_padding_03cm_live_check_01.json。
不要把实体回零记录里的0.05改写成0.03，也不要写成3cm配置已完成整轮实测。

13段放置规则
------------
垂直抬高 -> 拾取点上方调整姿态 -> 四段等高平移（3个中间点和盒上方终点）
-> 垂直下放 -> 松爪 -> 垂直上撤 -> 四段原路返回 -> 恢复空载姿态。
采用独立MoveIt Pilz LIN；平移段保持姿态，模型轨迹点及中点做碰撞/FK检查。
当前姿态来自离线选定的20°调整、世界XY轴315°方向候选；实际XYZ按本轮场景重算。
PLACED_AND_RETURNED只表示返回拾取侧高处并恢复姿态，不表示已回机械零位。
旧place-test保留pitch=0.7rad的放置方式；cycle不带--waypoints仍进入旧分支。

下一轮命令
----------
当前控制若仍保持，先确认方块已放回盒外桌面、场景固定、人员离开运动范围。
无需重复启动H1；查询：
  cd $HOME/openarm_vla_ws
  ./vision_grasp_baseline/run.sh status

确认state=idle、fault=null、synchronized=true、无pending后，实测入口为：
  ./vision_grasp_baseline/run.sh cycle --waypoints

该入口现在在开始及放置验证成功后都规划回零；新增末尾回零待完整实体测试。
新cycle自动创建独立runs目录，不能把历史place_retry_02命令直接再次粘贴运行。
只验证起始回零和抓取/抬升规划、不发送运动：
  ./vision_grasp_baseline/run.sh cycle --waypoints --plan-only
这里的--plan-only目前不包含完整13段放置预规划，不可把输出当作整轮证明。

控制后端已关闭时的启动
----------------------
使用独立实测位置保持入口，先保持，不直接发送机械零位目标：
终端1：
  cd $HOME/openarm_vla_ws
  ./vision_grasp_baseline/run.sh recovery-h1
等待左右臂JTC、right_gripper_controller和joint_state_broadcaster激活，保留终端。
该选项用于接管本轮曾测得的右肘-0.058rad姿态；任务轨迹仍逐点保留原-0.02rad下限。
回零轨迹不能比已测负向初始位置继续伸展；不修改原ROS包/ACT的全局关节限制。
终端2：
  cd $HOME/openarm_vla_ws
  ./vision_grasp_baseline/run.sh gateway
终端3：
  cd $HOME/openarm_vla_ws
  ./vision_grasp_baseline/run.sh synchronize-hold
  ./vision_grasp_baseline/run.sh status
同步是异步的；若显示draining，稍后再次查询status，等idle再执行cycle。
只有已定位的软件子程序异常且目标已清空、保持已验证时，reset-fault才可复位；
它不清除电机硬件故障。旧h1入口要求机械零位及支撑，不能代替非零姿态恢复入口。

停止本目录测试：
  ./vision_grasp_baseline/run.sh stop
它请求取消动作并保留ROS保持，不是回零、断电或实体急停。
关闭保持控制程序前先可靠支撑双臂。新cycle会清除旧STOP。

当前软件修复
------------
阶段交接等待：子程序退出后等待网关draining结束，避免立即申请下一阶段控制权。
场景恢复：附着ADD会移除同名世界物体，恢复夹持模型时不再重复发送世界REMOVE。
轨迹转换：ROS set_message_fields会修改输入列表，转换时传入深拷贝；
预规划后逐段检查相同的JSON到ROS目标准备流程，保存日志字典保持可序列化。
回零：zero-test绑定新鲜电机反馈，并在控制器返回后核对真实关节位置。

验证与证据
----------
原抓放软件回归22项；加入采集、临时附着及记录器修复后35项通过；另有本轮13段真实保存轨迹的ROS往返转换一致性检查。
复现软件检查（不运行机械臂）：
  source scripts/home_replay/workstation_env.sh
  source .control_ownership_overlay/install/setup.bash
  vision_grasp_baseline/.venv/bin/python -m pytest vision_grasp_baseline/tests -q

本轮主要证据位于runs/cycle_20261021_164451/：
  grasp/result.json                         抓取抬升、各段实际姿态及视觉验收
  place/result.json                         首次放置失败，尚未搬运
  place_retry_01/result.json                 JSON转换失败，physical_motion=false
  place_retry_02/result.json                 13段续接成功，10帧入盒验证
  place_retry_02/after_release/color_009.png  入盒照片
  return_zero_execution_01.json              单独实体回零，用5cm边缘余量
  return_zero_execution_01.raw_can.jsonl      本次回零原始命令/反馈
  table_padding_03cm_live_check_01.json       无运动刷新为3cm，并核验已在零位

历史first_physical_test_20261021.json、repeat_physical_test_20261021.json及旧runs保留。
旧轮碰桌、12.8cm撤离误差、J4的0xC故障属于历史事件，未覆盖或改写。
RESUME_WAYPOINTS_20261021.txt已归档为本轮续接说明，不再是待执行操作单。

标定、数据和实现边界
--------------------
全局相机D435i SDK序列号344322073972；640x480彩色与对齐深度。
使用SDK内参、深度尺度和Depth->Color外参；窄线缆分割不提供方块深度。
原标定config/global_camera_20261015.calib保持不变；本目录局部平移修正约
[+4.0,+1.6,+13.6]mm，见config/session_registration_20261021.json。
相机/底座移动后需要重新核验；历史碰撞尚未由旧几何模型可靠复现。
夹爪为原装OpenArm V2指尖。
历史runs是诊断与轨迹记录；新collect另外生成连续episodes，经审核和export_act转换后用于训练。
当前结果也不能证明ACT失败就是训练数据造成，或数据差异已完全排除。

官方实现参考（现有实现背景）
----------------------------
https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html
https://github.com/moveit/moveit2/blob/main/moveit_core/planning_scene/src/planning_scene.cpp
https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
