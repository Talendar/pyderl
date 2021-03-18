import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pyderl as drl
from pyderl.utils.gym_wrappers.atari import make_atari, wrap_deepmind
from timeit import default_timer as timer

ENV = make_atari("BreakoutNoFrameskip-v4")
ENV = wrap_deepmind(ENV, frame_stack=True, scale=True)


if __name__ == "__main__":
    # agent = drl.agents.RainbowDQNAgent(
    #     model=tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
    #         tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
    #         tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(512, activation="relu"),
    #         tf.keras.layers.Dense(ENV.action_space.n, activation="linear"),
    #     ]),
    #     num_actions=ENV.action_space.n,
    #     input_shape=ENV.observation_space.sample().shape,
    #     build_dueling=False,
    #     learning_rate=5e-4,
    #     replay_buffer_size=int(3e5),
    #     batch_size=32,
    # )
    input_stream_layers = [
        tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
        tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
        tf.keras.layers.Flatten(),
    ]

    agent = drl.agents.RainbowDQNAgent(
        input_shape=ENV.observation_space.sample().shape,
        num_actions=ENV.action_space.n,
        batch_size=32,
        input_stream_layers=input_stream_layers,
        build_dueling=True,
        adv_stream_hidden_units=256,
        state_stream_hidden_units=256,
        learning_rate=5e-4,
        replay_buffer_size=int(3e5),
    )
    agent.visualize()

    start_time = timer()
    history = agent.train(env=ENV,
                          total_timesteps=int(4e6),
                          exploration_fraction=0.2,
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
