"""
TODO
"""

import os
import pathlib
from collections import deque
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, List, Optional, Union, Tuple

import gym
import numpy as np
import tensorflow as tf
from PIL import Image

from pyderl.agents.base_agent import BaseAgent
from pyderl.agents.dqn.prioritized_replay_buffer import PrioritizedReplayBuffer
from pyderl.utils import LinearInterpolator
from pyderl.utils import compute_eta


class RainbowDQNAgent(BaseAgent):
    """ TODO

    .. todo::
        [X] Double Deep Q Learning
        [X] Prioritized Experience Replay
        [X] Dueling DQN Architecture
        [ ] A3C
        [ ] Distributional Q Learning
        [ ] Noisy DQN

    Args:
        network (tf.keras.Model): The keras model to be used by the agent.
        num_actions (int): Number of actions in the environment's action space.
            Currently, only environments with discrete action spaces are
            supported.
        gamma (float): Future reward discount factor.
        replay_buffer_size (int): Maximum number of transitions that can be
            stored at the same time in the replay buffer.
        prioritized_replay_alpha (float): Value of the alpha constant used by
            the prioritized experience replay buffer.
        prioritized_replay_epsilon (float): Value of the epsilon constant to
                be added to the TD errors when updating the buffer's priorities.
    """

    def __init__(self,
                 num_actions: int,
                 input_shape: Tuple[int, ...],
                 batch_size: int = 32,
                 model: Optional[tf.keras.Model] = None,
                 input_stream_layers: Optional[
                     List[tf.keras.layers.Layer]] = None,
                 build_dueling: bool = True,
                 state_stream_hidden_units: Union[int, List[int]] = 256,
                 adv_stream_hidden_units: Union[int, List[int]] = 256,
                 gamma: float = 0.99,
                 learning_rate: float = 5e-4,
                 replay_buffer_size: int = 100000,
                 prioritized_replay_alpha: float = 0.6,
                 prioritized_replay_epsilon: float = 1e-6) -> None:
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._num_actions = num_actions
        self._gamma = gamma
        self.prioritized_replay_epsilon = prioritized_replay_epsilon
        input_tensor = tf.keras.layers.Input(shape=input_shape,
                                             batch_size=batch_size)

        # Building dueling network:
        if build_dueling:
            # Assertions
            if input_stream_layers is None:
                raise ValueError("The layers of the input stream must be "
                                 "specified in order for the dueling network "
                                 "to be built!")
            if model is not None:
                raise ValueError("A model shouldn't be passed as argument when "
                                 "`build_dueling` is set to `True`!")

            # Converting int to list:
            if type(state_stream_hidden_units) == int:
                state_stream_hidden_units = [state_stream_hidden_units]

            if type(adv_stream_hidden_units) == int:
                adv_stream_hidden_units = [adv_stream_hidden_units]

            # Building the network:
            model = RainbowDQNAgent.build_dueling_architecture(
                inputs=input_tensor,
                num_actions=num_actions,
                input_stream_layers=input_stream_layers,
                state_stream_hidden_units=state_stream_hidden_units,
                adv_stream_hidden_units=adv_stream_hidden_units,
            )
        # Assertions
        else:
            if input_stream_layers is not None:
                raise ValueError("An input stream should be only passed as "
                                 "argument when `build_dueling` is set to"
                                 "`True`!")
            if model is None:
                raise ValueError("A full model must be specified when "
                                 "`build_dueling` is set to `False`!")

        # Building replay buffer:
        self.replay_buffer = PrioritizedReplayBuffer(
            size=replay_buffer_size,
            alpha=prioritized_replay_alpha,
        )

        # Online and target networks:
        self.online_model = model
        self.online_model(input_tensor)

        self.target_model = tf.keras.models.clone_model(self.online_model)
        self.update_target_network()

        # Building optimizer:
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate,
                                            clipnorm=1.0)

        # Loss function:
        self.loss = tf.keras.losses.Huber()

    @staticmethod
    def build_dueling_architecture(inputs: Any,
                                   num_actions: int,
                                   input_stream_layers: List[
                                       tf.keras.layers.Layer],
                                   state_stream_hidden_units: List[int],
                                   adv_stream_hidden_units: List[int],
    ) -> tf.keras.Model:
        """ TODO

        Args:
            inputs:
            num_actions:
            input_stream_layers:
            state_stream_hidden_units:
            adv_stream_hidden_units:

        Returns:

        """
        # Main layers:
        input_stream_out = inputs
        for layer in input_stream_layers:
            input_stream_out = layer(input_stream_out)

        # State value:
        state_value = input_stream_out
        for n, units in enumerate(state_stream_hidden_units):
            state_value = tf.keras.layers.Dense(
                units=units, activation="relu", name=f"state_dense{n}",
            )(state_value)
        state_value = tf.keras.layers.Dense(units=1,
                                            activation="linear",
                                            name="state_value")(state_value)

        # Advantages:
        advantages = input_stream_out
        for n, units in enumerate(adv_stream_hidden_units):
            advantages = tf.keras.layers.Dense(
                units=units, activation="relu", name=f"adv_dense{n}",
            )(advantages)
        advantages = tf.keras.layers.Dense(units=num_actions,
                                           activation="linear",
                                           name="advantages_raw")(advantages)
        advantages = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True),
            name="advantages_mean_sub",
        )(advantages)

        # Output:
        q_out = tf.keras.layers.Add(name="q_values")([state_value, advantages])

        return tf.keras.Model(inputs=inputs, outputs=q_out)

    def visualize(self) -> None:
        temp_file_name = ".temp_model_img.png"
        tf.keras.utils.plot_model(self.online_model,
                                  to_file=temp_file_name,
                                  show_shapes=True)

        Image.open(temp_file_name).show()
        os.remove(temp_file_name)

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

    def update_target_network(self) -> None:
        """ Updates the weights of the target network. """
        self.target_model.set_weights(self.online_model.get_weights())

    def _experience_replay(self, obses_t, actions, rewards,
                           obses_tp1, dones, prb_weights, batch_idxes):
        # Best actions for the next observations:
        q_tp1_online = self.online_model(obses_tp1)
        q_tp1_online_best = tf.argmax(q_tp1_online, axis=1)
        q_tp1_online_best_mask = tf.one_hot(indices=q_tp1_online_best,
                                            depth=self._num_actions)

        # Target Q-values:
        q_tp1_target = self.target_model(obses_tp1)
        q_tp1_target_best = tf.boolean_mask(q_tp1_target,
                                            q_tp1_online_best_mask)
        q_tp1_target_best = q_tp1_target_best * (1 - dones)
        q_t_true = rewards + self._gamma * q_tp1_target_best

        # Mask for the actions selected in the current transitions:
        current_actions_mask = tf.one_hot(indices=actions,
                                          depth=self._num_actions)

        # Predicting online Q-values and calculating loss:
        with tf.GradientTape() as tape:
            q_t_online = self.online_model(obses_t)
            q_t_pred = tf.boolean_mask(q_t_online, current_actions_mask)
            assert q_t_pred.shape == q_t_true.shape

            loss = self.loss(y_true=q_t_true, y_pred=q_t_pred)
            weighted_loss = tf.reduce_mean(prb_weights * loss)

        # Gradient descent step:
        gradients = tape.gradient(weighted_loss,
                                  self.online_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.online_model.trainable_variables)
        )

        # Updating prioritized replay buffer:
        # todo: solve repeated calculation of td_error (already calculated by
        #  the loss function)
        abs_td_errors = np.abs(q_t_pred - q_t_true)
        new_priorities = abs_td_errors + self.prioritized_replay_epsilon
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def train(self,
              env: gym.Env,
              total_timesteps: int = 100000,
              exploration_fraction: float = 0.2,
              min_exploration_chance: float = 0.02,
              train_freq: int = 4,
              learning_starts_step: int = 1000,
              target_network_update_freq: int = 500,
              prioritized_replay_beta0: float = 0.4,
              prioritized_replay_beta_fraction: float = 1.0,
              checkpoint: bool = False,
              checkpoint_path: Optional[str] = None,
              checkpoint_freq: int = 1000,
              verbose: bool = True,
              render_env: bool = False) -> List[float]:
        """ TODO

        Args:
            env (gym.Env): Environment in which the agent will be trained on.
            total_timesteps (int): Total number of steps to be taken on the
                given environment.
            exploration_fraction (float): Fraction/percentage of the training
                period over which the exploration rate is annealed.
            min_exploration_chance (float): Final value of the exploration rate.
            train_freq (int): A training session, in which the model's
                parameters are updated, is performed every `train_freq` steps.
            learning_starts_step (int): Amount of steps to wait before starting
                regular training sessions. During this period, the agent is only
                collecting transitions from the environment.
            target_network_update_freq (int): The update frequency of the target
                network.
            prioritized_replay_beta0 (float): Initial value of the beta variable
                used by the priority replay buffer.
            prioritized_replay_beta_fraction (float): Fraction/percentage of the
                training period over which the beta variable is annealed.
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

        # Linear interpolator for the prioritized replay buffer beta variable:
        prb_beta = LinearInterpolator(
            initial_value=prioritized_replay_beta0,
            final_value=1.0,
            num_timesteps=int(
                total_timesteps * prioritized_replay_beta_fraction
            ),
        )

        # Storage for the total rewards obtained in each episode:
        episodes_rewards = [0.0]

        # Storage for the time taken to perform the last n steps:
        steps_times = deque(maxlen=2000)

        # Training:
        obs = env.reset()
        for t in range(total_timesteps):
            # Starting timer:
            step_start_time = timer()

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
                    avg_reward = np.mean(episodes_rewards
                                         if len(episodes_rewards) <= 100
                                         else episodes_rewards[-100:])
                    avg_time_per_step = np.mean(steps_times)
                    eta = compute_eta(avg_time_per_step=avg_time_per_step,
                                      remaining_steps=(total_timesteps - t))
                    pc_completed = (t + 1) / total_timesteps

                    print(
                        f"[{pc_completed:.2%}]"
                        f"[E: {len(episodes_rewards)}]"
                        f"[T: {t + 1}/{total_timesteps}] "
                        f"Reward: {episodes_rewards[-1]}"
                        f"  |  "
                        f"Avg. reward (100 past episodes): {avg_reward:.2f}"
                        f"  |  "
                        f"Exploration chance: {exploration_chance.value(t):.2%}"
                        f"\nETA: {eta}  |  "
                        f"Avg. time per step: {1e3 * avg_time_per_step:.2f}ms\n"
                    )

                # Resetting:
                obs = env.reset()
                episodes_rewards.append(0.0)

            # Experience replay:
            if t >= learning_starts_step and (t % train_freq) == 0:
                experience = self.replay_buffer.sample(
                    batch_size=self._batch_size,
                    beta=prb_beta.value(t),
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

            # Storing the amount of time taken to complete the step:
            steps_times.append(timer() - step_start_time)

        return episodes_rewards
