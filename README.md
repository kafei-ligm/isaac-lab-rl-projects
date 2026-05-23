# Isaac Lab 强化学习项目集

基于 NVIDIA Isaac Lab 平台的机器人强化学习实验项目，涵盖运动控制与操作任务。

## 环境

- Isaac Lab 2.3.2 + Isaac Sim 5.1.0
- GPU: NVIDIA RTX 4060 Laptop (8GB)
- 算法: PPO (rsl-rl-lib 3.0.1) / SAC (skrl 2.0.0)

## 项目列表

### [Project 01: H1人形机器人 Locomotion Reward优化与课程学习](project_01_locomotion/)

使用 **PPO** 训练宇树 H1 人形机器人平地行走，通过系统性的 reward 调优实验，将总奖励提升 **68%**，实现零摔倒。并验证课程学习在复杂地形上的迁移效果，reward **由负转正**。

**关键发现**：reward 各项权重的平衡比单项权重的大小更重要。

### [Project 02: Franka机械臂 Open Drawer — SAC算法调优与域随机化](project_02_manipulation/)

使用 **SAC** 训练 Franka Panda 机械臂完成开抽屉任务。系统研究了 entropy 超参数的影响（最优值 0.05），并通过域随机化发现关节随机化反而提升性能至 **90.99**（超越 baseline 的 80.16）。

**关键发现**：SAC 的 entropy 存在最优平衡点；状态空间随机化可以弥补低 entropy 的探索不足。

## 技术栈

| 项目 | 算法 | 机器人 | 任务 | 框架 |
|------|------|--------|------|------|
| Project 01 | PPO | 宇树 H1 人形机器人 | Locomotion | rsl-rl-lib |
| Project 02 | SAC / PPO | Franka Panda 机械臂 | Open Drawer | skrl / rsl-rl-lib |

## 作者

（李佳霏/GitHub ID）
