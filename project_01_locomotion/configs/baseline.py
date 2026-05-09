"""Baseline配置：IsaacLab默认H1 flat环境参数"""

# 关键reward权重（默认值）
REWARD_CONFIG = {
    "track_lin_vel_xy_exp": 1.0,    # 线速度跟踪
    "track_ang_vel_z_exp": 1.0,     # 转向跟踪（默认）
    "termination_penalty": -200.0,  # 摔倒惩罚
    "feet_air_time": 0.25,          # 步态奖励
    "feet_slide": -0.25,            # 脚滑动惩罚
    "dof_pos_limits": -1.0,         # 关节限位惩罚
    "joint_deviation_hip": -0.2,    # 髋关节偏差惩罚
    "joint_deviation_arms": -0.2,   # 手臂偏差惩罚
    "joint_deviation_torso": -0.1,  # 躯干偏差惩罚
    "flat_orientation_l2": -1.0,    # 姿态惩罚
    "action_rate_l2": -0.005,       # 动作平滑惩罚
    "dof_acc_l2": -1.25e-7,         # 关节加速度惩罚
}

# 训练结果
RESULTS = {
    "mean_reward": 24.32,
    "episode_length": 973.30,
    "error_vel_xy": 0.3200,
    "error_vel_yaw": 0.6171,
    "fall_rate": "3.1%",
}