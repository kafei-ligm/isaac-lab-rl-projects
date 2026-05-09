# Isaac Lab 强化学习项目集

基于 NVIDIA Isaac Lab 平台的机器人强化学习实验项目，涵盖运动控制与操作任务。

## 环境

- Isaac Lab 2.3.2 + Isaac Sim 5.1.0
- GPU: NVIDIA RTX 4060 Laptop (8GB)
- 算法: PPO / SAC (rsl-rl-lib)

## 项目列表

### [Project 01: H1人形机器人 Locomotion Reward优化](project_01_locomotion/)

使用 PPO 训练宇树 H1 人形机器人平地行走，通过系统性的 reward 调优实验，将总奖励提升 68%，实现零摔倒。

**关键发现**：reward shaping 中各项权重的平衡比单项权重的大小更重要。

### Project 02: 机械臂抓取任务（计划中）

使用 SAC 算法训练机械臂完成抓取/操作任务。

## 作者

（你的名字/GitHub ID）
