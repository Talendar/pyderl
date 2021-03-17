import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pyderl as drl
from timeit import default_timer as timer

ENV = gym.make("CartPole-v0")


if __name__ == "__main__":
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(ENV.action_space.n, activation="linear")
    ])
    network.build(input_shape=[None, *ENV.observation_space.sample().shape])

    agent = drl.agents.RainbowDQNAgent(network=network,
                                       num_actions=ENV.action_space.n,
                                       learning_rate=5e-4,
                                       replay_buffer_size=int(1e6))

    start_time = timer()
    history = agent.train(env=ENV,
                          total_timesteps=32000,
                          exploration_fraction=0.6,
                          render_env=True)
    deltaT = timer() - start_time
    print(f"Training time: {deltaT / 60:.2f} min")

    interval = 50
    plt.plot(range(interval, len(history)),
             [np.mean(history[(i - interval):i])
              for i in range(interval, len(history))])
    plt.show()

    while True:
        opt = input("\nPress 0 to exit or any other key to visualize the "
                    "agent.\n")
        try:
            opt = int(opt)
            if opt == 0:
                break
        except ValueError:
            pass

        drl.utils.visualize_agent(agent, ENV, fps=80)
