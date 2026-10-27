> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# 下一步行动清单（修改于 2026-09-11）

- 建立日期：2026-09-03
- 修改日期：2026-09-11
- 当前权威状态：[蓝色方块固定黑盒：ACT 200轨迹、50k训练与30-seed闭环](../当前阶段/蓝色方块_固定黑盒_ACT_200轨迹_50k训练与30seed闭环_建立2026-09-11_修改2026-09-11.md)

## 当前状态

- 固定黑盒专家数据：216 attempts、200 online success、严格 replay `200/200`。
- 固定切分：160 train / 40 holdout；两份raw和LeRobot完整验证通过。
- 黑盒LeRobot训练集：160 episodes、185,890 frames。
- 黑盒ACT：50,000 steps正常完成，final loss `0.026`。
- 黑盒50k未见seed闭环：真实落盒 `27/30=90.0%`，完整回位同为 `27/30=90.0%`。
- 黑盒失败：`lift_hold_condition_not_met` 2条，`placement_condition_not_met` 1条。

以下条目是上一阶段附近随机放置基线：

- 旧橙色 ACT 路线：已证明数据转换、训练、checkpoint reload 和 MuJoCo 闭环链路可行。
- 蓝色完整任务专家数据：最终200条严格有效，固定切分160 train / 40 holdout。
- 新增批次：120 online success，严格 replay `118/120`。
- 蓝色 LeRobot 数据：160 episodes、111,275 frames，完整验证通过。
- 蓝色 ACT：20,000 steps 正常结束；最终 train loss `0.056`，eval loss `0.0371`。
- 旧20k中央范围抓取保持：`13/30=43.3%`。
- 新50k训练：50,000 steps正常完成，final loss `0.036`。
- 新50k完整范围物体任务：`26/30=86.7%`；学习式回位 `0/26`。

## P0：比较黑盒40k/45k/50k checkpoint

1. 对 `040000`、`045000`、`050000` 使用同一批 `20273000..20273029`。
2. 使用 `openarm_cube_into_black_box_v3` 成功合同，禁止使用旧水平位移判定。
3. 按 `task_success` 和 `full_sequence_success` 排名；当前50k基线均为 `27/30`。
4. 保存每个seed独立JSON，并冻结胜出checkpoint的模型SHA256。

## P1：失败可视化和分桶

- `20273003`：抬升后未连续稳定保持2 s；
- `20273019`：释放后落在盒外；
- `20273027`：没有形成双指接触；
- 按初始X、Y、yaw和失败阶段分桶，重点关注右侧边缘与较大正yaw。

## P2：100个以上新seed压测

- checkpoint选择完成后，再使用与训练和30-seed都不重叠的新seed；
- 报告真实盒内稳定、回位、失败类型和每个分桶成功率；
- 目标是确认90%不是30个seed的小样本波动。

## 历史P0：附近随机放置可视化

1. 使用50k checkpoint：

   ```text
   outputs/act_openarm_cube_pick_blue_200_50k_20260908/checkpoints/050000/pretrained_model
   ```

2. 已使用与采集一致的 `X±0.05 m / Y±0.06 m` 起点范围。
3. 物体任务成功 `26/30=86.7%`。
4. 使用MuJoCo viewer检查成功seed `20268000`、抓取失败seed `20268003` 和放置失败seed `20268018`。
5. viewer闭环控制45秒，最终画面保留60秒；回位作为独立复位指标。

## 历史P1：比较附近随机放置 checkpoint

- 对5k、10k、15k直到50k使用同一批未见seed。
- 汇总字段使用 `task_success`，保存每个seed的独立JSON。
- `task_success` 必须覆盖抓取、保持2秒、放置和松爪；回位由
  `full_sequence_success` 单独记录。
- 选择闭环成功率最高的checkpoint，不默认选择最终50k。

## 历史P2：依据附近放置闭环结果决策

- 若物体任务成功率达到预设门槛：冻结数据、checkpoint、验证器和30-seed报告，再进入更多seed压测。
- 若能抓取但不能放置：重点检查 chunk 长时序、HOLD 后状态分布和 PLACE/RELEASE 标签。
- 若接近方块但抓取失败：按起始 XY、yaw、抓取候选和相机视角分桶分析。
- 若机械臂动作异常：先检查 state normalization、action 解归一化和 executor 调度，不以 loss 下降代替闭环诊断。

## P3：进入 π0 前的门禁

只有在蓝色完整任务的专家数据、ACT 训练和未见 seed 闭环报告均被冻结后，再开始
π0/π0.5 云端训练测试。ACT 是当前数据/控制/评估链的基线，不应跳过其失败分析。

## 禁止误用

- 不再引用旧 `g5_1000_seed_20260902.json` 代表当前蓝色模型。
- 不把 20k 离线 eval loss 当作完整任务成功。
- 不把专家在线 success、严格 replay valid、LeRobot episode 数和 ACT train/eval split 混为同一数字。
- 不把旧验证器的抓取保持成功率写成完整任务成功率。
- 不把旧附近随机放置 `26/30` 与固定黑盒 `27/30` 当成同口径直接排名。
- 固定黑盒成功必须检查 `inside_black_box` 和至少0.5 s稳定窗口。
