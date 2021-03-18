""" Utility functions.
"""

import time

import gym

from pyderl.agents.base_agent import BaseAgent


def visualize_agent(agent: BaseAgent,
                    env: gym.Env,
                    num_episodes: int = 1,
                    fps: int = 60) -> None:
    episodes_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episodes_rewards.append(0.0)

        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            episodes_rewards[-1] += reward

            env.render()
            time.sleep(1 / fps)

        print(f"Episode reward: {episodes_rewards[-1]}")

    print(f"Total reward: {sum(episodes_rewards)}")


def compute_eta(avg_time_per_step: float,
                remaining_steps: int) -> str:
    """ todo """
    remaining_time = int(avg_time_per_step * remaining_steps)
    m, s = divmod(remaining_time, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
