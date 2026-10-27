> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# OpenArm VLA 文档索引（修改于 2026-09-11）

- 建立日期：2026-09-03
- 修改日期：2026-09-11

## 当前权威入口

- [蓝色方块固定黑盒：ACT 200轨迹、50k训练与30-seed闭环](当前阶段/蓝色方块_固定黑盒_ACT_200轨迹_50k训练与30seed闭环_建立2026-09-11_修改2026-09-11.md)：最新任务权威状态，真实落盒并回位 `27/30=90.0%`。
- [蓝色方块200轨迹、ACT 50k训练与闭环验证](当前阶段/蓝色方块_ACT_200轨迹_50k训练与闭环验证_建立2026-09-09_修改2026-09-09.md)：上一阶段附近随机放置权威快照与MuJoCo可视化命令。
- [蓝色方块 200 轨迹与 ACT 50k 训练前状态](当前阶段/蓝色方块_ACT_200轨迹与50k训练前状态_建立2026-09-08_修改2026-09-08.md)：上一阶段数据和训练准备快照。
- [蓝色方块 100 轨迹与 ACT 20k 状态报告](当前阶段/蓝色方块_ACT_100轨迹与20k训练状态_建立2026-09-08_修改2026-09-08.md)：上一阶段基线与历史训练结果。
- [下一步行动清单](历史任务文档/NEXT_STEPS.md)：当前未完成门禁和执行顺序。
- [工作空间组织](工作空间组织/WORKSPACE_ORGANIZATION.md)：代码、数据、模型和结果目录说明。
- [ACT 训练前门禁](../docs/ACT_TRAINING_PREP.md)：原始数据、严格回放和 normalization 门禁。

## 当前结论（2026-09-11）

当前最新任务已经从“附近随机放置”升级为固定可见黑盒：

```text
随机蓝色方块（X±0.05 m / Y±0.04 m / yaw±15°）
→ 抓取并保持2 s
→ 搬运至固定黑盒
→ 松爪并在盒内稳定0.5 s
→ 右臂回位，左臂保持停放状态
```

黑盒任务已确认：

1. 专家采集216次，得到200条online success，成功率92.59%；
2. 200/200全部通过严格actuator replay；
3. 固定切分160 train / 40 holdout，两份raw和LeRobot验证均通过；
4. LeRobot训练集160 episodes、185,890 frames、双路960×600视频；
5. ACT 50k训练用时2:51:47，final loss `0.026`；
6. 50k checkpoint在30个完全未见seed上真实落盒并回位 `27/30=90.0%`；
7. 三条失败为2条抓取/稳定保持失败和1条落盒失败，没有成功落盒后的回位失败。

当前模型：

```text
outputs/act_openarm_cube_into_black_box_200_50k_20260910/
```

50k模型SHA256：

```text
bbf2516f2d91c34c1c801826be834cf2920b49254dfb79be4fbb62c703686cef
```

以下内容是上一阶段“附近随机放置”基线，仅用于对照。

项目已经从旧的橙色方块“仅抓取”验证进入蓝色方块完整任务：

```text
随机蓝色方块
→ 右臂抓取
→ 抬升至少 5 cm
→ 稳定保持 2 s
→ 搬运到附近随机目标
→ 下放并松爪
→ 安全撤离
→ 右臂回到初始姿态
```

当前已确认：

1. 旧80条训练集的 ACT 20k 模型在30个未见 seed 上成功 `13/30 = 43.3%`。
2. 新批次采集175次，获得120条在线成功轨迹；严格 replay 为 `118/120`。
3. 最终严格有效数据为200条，固定切分为160条训练和40条holdout。
4. 新 LeRobot v3训练集包含160 episodes、111,275 frames和两路960×600视频。
5. 完整遍历验证结果为 `valid=true`、`errors=[]`，使用 STATE=IDENTITY。
6. ACT 50k训练已完成，用时2:48:29，final loss `0.036`，共保存10个checkpoint。
7. 完整范围30个全新seed中，50k完成抓取保持27、松爪27、稳定放置26，物体任务
   成功 `26/30=86.7%`；学习式回位为 `0/26`，作为独立复位指标记录。

## 上一阶段模型（附近随机放置）

最终 checkpoint：

```text
outputs/act_openarm_cube_pick_blue_100_20k/
  checkpoints/020000/pretrained_model/
```

`model.safetensors` SHA256：

```text
87b3c4adcaf282a18d55cd00c442f866ae1c46dc59a513e16c891aeec41abf49
```

新50k输出目标：

```text
outputs/act_openarm_cube_pick_blue_200_50k_20260908/
```

当前状态：50k训练完成；50k模型SHA256为
`661f4e33c036414878e8ce6e0ad23fe04dac7d4e17ad57a2a09dc186ebe2b0fd`。

## 历史文档

`MD/历史任务文档/` 中带旧日期的验收报告是当时状态快照，用于追溯，不代表
2026-09-11 的最新结论。旧橙色任务的 G5/G5.5 结论不得覆盖当前固定黑盒任务的
数据与模型结论。

## 当前禁止误用

- 当前固定黑盒 checkpoint 必须指向
  `outputs/act_openarm_cube_into_black_box_200_50k_20260910/checkpoints/050000/pretrained_model`。
- 不把在线采集 `100/100` 等同于严格回放 `100/100`；实际主批次为 `96/100`。
- 不把旧 LeRobot 80 episodes 与新 LeRobot 160 episodes 混用。
- 不把离线 `eval_loss=0.0371` 当成 MuJoCo 闭环成功率。
- 不使用 checkpoint 上一级目录加载策略；应指向 `020000/pretrained_model`。
- 汇总闭环结果读取 `task_success`，不是不存在的 `success` 字段。
- 完整任务成功必须同时覆盖抓取、2 秒保持、放置、松爪和回位。
- 旧20k的 `13/30=43.3%` 和50k的 `26/30=86.7%` 是旧中央范围抓取保持对照；
  新完整范围物体任务结果也为 `26/30=86.7%`，两种测试口径不能混写。
- 旧附近放置范围为 `X±0.05 m / Y±0.06 m`；固定黑盒范围为
  `X±0.05 m / Y±0.04 m`。
- `task_success` 表示抓取、保持、放置、松爪；`full_sequence_success` 额外要求回位。
