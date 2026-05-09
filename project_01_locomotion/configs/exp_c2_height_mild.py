"""实验C2：加入躯干高度惩罚（适中权重 -0.2）— 最佳配置"""

# 改动说明：
# 发现C1的height权重-1.0过强，降低到-0.2
# 结果：所有实验中最佳，reward提升68%，零摔倒

CHANGES = {
    "track_ang_vel_z_exp": "1.0 → 2.0",
    "base_height": "新增，weight=-0.2, target_height=0.98",
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
    "base_height": -0.2,           # ← 新增（适中）
}

RESULTS = {
    "mean_reward": 40.93,
    "episode_length": 991.79,
    "error_vel_xy": 0.3065,
    "error_vel_yaw": 0.5037,
    "fall_rate": "0%",
    "note": "最佳配置：reward最高，零摔倒，姿态良好",
}