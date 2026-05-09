"""实验C1：加入躯干高度惩罚（权重过强 -1.0）"""

# 改动说明：
# 在实验B基础上，新增base_height_l2惩罚项
# 目标高度0.98m，权重-1.0
# 结果：权重过强导致性能严重下降

CHANGES = {
    "track_ang_vel_z_exp": "1.0 → 2.0",
    "base_height": "新增，weight=-1.0, target_height=0.98",
}

REWARD_CONFIG = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 2.0,
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
    "base_height": -1.0,           # ← 新增（过强）
}

RESULTS = {
    "mean_reward": 25.13,
    "episode_length": 918.61,
    "error_vel_xy": 0.3642,
    "error_vel_yaw": 0.7328,
    "fall_rate": "20.1%",
    "note": "权重过大导致摔倒率飙升，性能全面下降",
}