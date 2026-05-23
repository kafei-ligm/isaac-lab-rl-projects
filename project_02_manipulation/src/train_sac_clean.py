"""SAC training script for Isaac Lab using skrl."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC with skrl.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=36000, help="Max training timesteps.")
parser.add_argument("--experiment_name", type=str, default="sac_baseline", help="Experiment name.")
parser.add_argument("--domain_rand", type=str, default="none", choices=["none", "wide_friction", "wide_joints"],
                    help="Domain randomization: none, wide_friction (wider friction range), wide_joints (wider joint reset).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.agents.torch.sac import SAC
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# Define the policy (actor) model
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std, reduction="sum")

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return torch.tanh(self.net(inputs["observations"])), {"log_std": self.log_std_parameter}


# Define the critic model
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)), {}


def main():
    """Train SAC agent."""
    from isaaclab.utils import configclass
    from isaaclab.envs.mdp import randomize_rigid_body_material, reset_joints_by_offset

    # parse env config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # === 域随机化 ===
    if args_cli.domain_rand == "wide_friction":
        # 扩大摩擦系数随机化范围（默认 0.8-1.25 → 0.4-2.0）
        env_cfg.events.robot_physics_material = EventTerm(
            func=randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.4, 2.0),
                "dynamic_friction_range": (0.4, 2.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        env_cfg.events.cabinet_physics_material = EventTerm(
            func=randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
                "static_friction_range": (0.4, 2.0),
                "dynamic_friction_range": (0.4, 2.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        print("[INFO] Domain randomization: wide friction (0.4-2.0)")

    elif args_cli.domain_rand == "wide_joints":
        # 扩大关节初始位置随机范围（默认 ±0.1 → ±0.3）
        env_cfg.events.reset_robot_joints = EventTerm(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.3),
                "velocity_range": (0.0, 0.0),
            },
        )
        print("[INFO] Domain randomization: wide joint reset (±0.3)")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    device = env.device
    set_seed(args_cli.seed)

    # instantiate models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    # replay buffer
    memory = RandomMemory(memory_size=100000, num_envs=env.num_envs, device=device)

    # SAC config
    cfg = {
        "batch_size": 256,
        "discount_factor": 0.99,
        "polyak": 0.005,
        "learning_rate": 3e-4,
        "random_timesteps": 1000,
        "learning_starts": 1000,
        "grad_norm_clip": 1.0,
        "learn_entropy": True,
        "initial_entropy_value": 0.05,
        "target_entropy": None,
        "rewards_shaper": None,
        "experiment": {
            "directory": "logs/skrl_sac",
            "experiment_name": args_cli.experiment_name,
            "write_interval": 100,
            "checkpoint_interval": 5000,
        },
    }

    # create SAC agent
    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # create trainer
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args_cli.max_iterations, "headless": True},
    )

    print(f"[INFO] Starting SAC training for {args_cli.max_iterations} timesteps")
    print(f"[INFO] Num envs: {env.num_envs}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Domain randomization: {args_cli.domain_rand}")

    start_time = time.time()
    trainer.train()
    print(f"[INFO] Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()