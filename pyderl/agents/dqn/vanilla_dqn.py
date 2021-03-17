""" Implementation of the vanilla Deep Q Network.
"""

import pathlib
from datetime import datetime
from typing import List, Optional, Union

import gym
import numpy as np
import tensorflow as tf

from pyderl.agents.base_agent import BaseAgent
from pyderl.agents.dqn.replay_buffer import ReplayBuffer
from pyderl.utils.linear_interpolator import LinearInterpolator


class DQNAgent(BaseAgent):
    """ Implementation of the original/vanilla version of the Deep Q Network, as
    described by :cite:`mnih:2015`.
    """

    def __init__(self,
                 network: tf.keras.Model,
                 num_actions: int,
                 gamma: float = 0.99,
                 learning_rate: float = 5e-4,
                 replay_buffer_size: int = 100000) -> None:
        self._num_actions = num_actions
        self._gamma = gamma

        self.replay_buffer = ReplayBuffer(
            input_shape=network.input_shape[1:],
            size=replay_buffer_size,
        )

        self.online_model = network
        self.target_model = tf.keras.models.clone_model(self.online_model)
        self.update_target_network()

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate,
                                            clipnorm=1.0)
        self.loss = tf.keras.losses.Huber()

    def act(self,
            obs: Union[np.ndarray, tf.Tensor],
            exploration_chance: float = 0.01) -> int:
        """ Returns an action chosen according to the agent's policy.

        Note:
            This method doesn't accept a batch of observations as input, just a
            single observation!

        Args:
            obs (Union[np.ndarray, tf.Tensor]): Observation of the environment's
                current state.
            exploration_chance (float): The probability of the agent choosing
                a random action instead of following its greedy policy.

        Returns:
            The action chosen by the agent.
        """
        # Exploring:
        if np.random.uniform(low=0, high=1) < exploration_chance:
            action = np.random.choice(self._num_actions)
        # Exploiting:
        else:
            # Preparing the input observation:
            obs_tensor = tf.convert_to_tensor(obs)
            obs_tensor = tf.expand_dims(obs_tensor, axis=0)

            # Predicting the Q values associated with each action:
            q_values = self.online_model(obs_tensor, training=False)

            # Selecting the optimal action:
            action = tf.argmax(q_values[0]).numpy()

        return action

    def _experience_replay(self, obs, actions,
                           rewards, next_obs, dones) -> None:
        """ Triggers a new learning session. """
        # Predicting target Q-values:
        q_target = tf.reduce_max(self.target_model(next_obs), axis=1)
        q_target = q_target * (1 - dones)  # set Q = 0 when the episode is done
        q_target = rewards + self._gamma * q_target

        # Predicting online Q-values and calculating loss:
        actions_mask = tf.one_hot(indices=actions, depth=self._num_actions)
        with tf.GradientTape() as tape:
            q_values = self.online_model(obs)
            q_action = tf.reduce_sum(
                tf.multiply(q_values, actions_mask), axis=1,
            )

            assert q_action.shape == q_target.shape
            loss = self.loss(y_true=q_target, y_pred=q_action)

        # Gradient descent step:
        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.online_model.trainable_variables)
        )

    def update_target_network(self) -> None:
        """ Updates the weights of the target network. """
        self.target_model.set_weights(self.online_model.get_weights())

    def train(self,
              env: gym.Env,
              total_timesteps: int = 100000,
              exploration_fraction: float = 0.3,
              min_exploration_chance: float = 0.05,
              train_freq: int = 4,
              batch_size: int = 32,
              learning_starts_step: int = 1000,
              target_network_update_freq: int = 500,
              checkpoint: bool = False,
              checkpoint_path: Optional[str] = None,
              checkpoint_freq: int = 1000,
              verbose: bool = True,
              render_env: bool = False) -> List[float]:
        """ Trains the agent on the given environment.

        Args:
            env (gym.Env): Environment in which the agent will be trained on.
            total_timesteps (int): Total number of steps to be taken on the
                given environment.
            exploration_fraction (float): Fraction/percentage of the training
                period over which the exploration rate is annealed.
            min_exploration_chance (float): Final value of the exploration rate.
            train_freq (int): A training session, in which the model's
                parameters are updated, is performed every `train_freq` steps.
            batch_size (int): Size of the batch sampled from the replay buffer
                during a training session.
            learning_starts_step (int): Amount of steps to wait before starting
                regular training sessions. During this period, the agent is only
                collecting transitions from the environment.
            target_network_update_freq (int): The update frequency of the target
                network.
            checkpoint (bool): Whether or not to save the agent's model
                periodically during the training.
            checkpoint_path (Optional[str]): Path to the directory in which the
                checkpoints will be stored. If `None`, a new directory with a
                default name will be created and used.
            checkpoint_freq (int): Frequency of the checkpoints.
            verbose (bool): Whether or not to print the training progress.
            render_env (bool): Whether or not to render the environment during
                the training.

        Returns:
            A list containing the rewards obtained by the agent on each episode.
        """
        # Creating checkpoint directory:
        if checkpoint:
            if checkpoint_path is None:
                t = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                checkpoint_path = f"./dqn_checkpoints_{t}"

            pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # Linear interpolator for the exploration chance:
        exploration_chance = LinearInterpolator(
            initial_value=1,
            final_value=min_exploration_chance,
            num_timesteps=int(total_timesteps * exploration_fraction),
        )

        # Storage for the total rewards obtained in each episode:
        episodes_rewards = [0.0]

        # Training:
        obs = env.reset()
        for t in range(total_timesteps):
            # Rendering env:
            if render_env:
                env.render()

            # Choosing action and updating env:
            action = self.act(obs, exploration_chance.value(t))
            next_obs, reward, done, _ = env.step(action)

            # Storing the transition into the replay buffer:
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # Checking if the episode is done:
            episodes_rewards[-1] += reward
            if done:
                # Printing info:
                if verbose:
                    # TODO: ETA based on avg time per step
                    avg_reward = np.mean(episodes_rewards
                                         if len(episodes_rewards) <= 50
                                         else episodes_rewards[-50:])
                    print(
                        f"[Episode {len(episodes_rewards)}]"
                        f"[Timestep {t + 1}/{total_timesteps}] "
                        f"Reward: {episodes_rewards[-1]}  |  "
                        f"Avg. reward (50 past episodes): {avg_reward:.2f}  |  "
                        f"Exploration chance: {exploration_chance.value(t):.2%}"
                    )

                # Resetting:
                obs = env.reset()
                episodes_rewards.append(0.0)

            # Experience replay:
            if t >= learning_starts_step and (t % train_freq) == 0:
                experience = self.replay_buffer.make_batch(
                    min(batch_size, len(self.replay_buffer))
                )
                self._experience_replay(*experience)

            # Updating target network:
            if (t >= learning_starts_step
                    and (t % target_network_update_freq) == 0):
                self.update_target_network()

            # Checkpoint
            if checkpoint and (t % checkpoint_freq) == 0:
                # TODO
                raise NotImplementedError()

        return episodes_rewards
