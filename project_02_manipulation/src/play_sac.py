"""SAC play script for Isaac Lab using skrl."""

import argparse
import time as t
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play SAC with skrl.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.agents.torch.sac import SAC
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return torch.tanh(self.net(inputs["observations"])), {"log_std": self.log_std_parameter}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)), {}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)
    device = env.device

    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "critic_1": Critic(env.observation_space, env.action_space, device),
        "critic_2": Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }

    # Load checkpoint
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    for name, model in models.items():
        if name in ckpt:
            model.load_state_dict(ckpt[name])
            print(f"[INFO] Loaded {name}")

    # Move to device and set to eval mode
    for model in models.values():
        model.to(device)
        model.eval()

    # Run playback
    obs, _ = env.reset()
    print("[INFO] Playing... Press Ctrl+C to stop")
    try:
        for step in range(10000):
            with torch.no_grad():
                observations = obs["policy"] if isinstance(obs, dict) else obs
                mean_actions = models["policy"].net(observations)
                actions = torch.tanh(mean_actions)
            obs, rewards, terminated, truncated, info = env.step(actions)
            t.sleep(0.02)
            if step % 100 == 0:
                print(f"Step {step}, Reward: {rewards.mean().item():.4f}")
    except KeyboardInterrupt:
        print("[INFO] Stopped by user")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
