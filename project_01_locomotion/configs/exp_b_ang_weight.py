"""实验B：提高转向跟踪权重 1.0 → 2.0"""

# 改动说明：
# 发现baseline转向误差较大(0.62)，分析原因是转向reward权重不足
# 将track_ang_vel_z_exp权重从1.0提高到2.0

CHANGES = {
    "track_ang_vel_z_exp": "1.0 → 2.0",  # 唯一改动
}

REWARD_CONFIG = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 2.0,     # ← 改动点
    "termination_penalty": -200.0,
    "feet_air_time": 0.25,
    "feet_slide": -0.25,
    "dof_pos_limits": -1.0,
    "joint_deviation_hip": -0.2,
    "joint_deviation_arms": -0.2,
    "joint_deviation_torso": -0.1,
    "flat_orientation_l2": -1.0,
    "action_rate_l2": -0.005,
    "dof_acc_l2": -1.25e-7,
}

RESULTS = {
    "mean_reward": 39.17,
    "episode_length": 990.48,
    "error_vel_xy": 0.2177,
    "error_vel_yaw": 0.4932,
    "fall_rate": "3.1%",
}