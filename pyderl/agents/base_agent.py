""" Defines a base abstract class for PyDeRL's agents.
"""

import abc
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy
    import tensorflow


class BaseAgent(abc.ABC):
    """ Base abstract class that defines an interface to PyDeRL's agents.

    .. note::
        Currently, the agents support only environments with a discrete action
        space.
    """

    @abc.abstractmethod
    def act(self,
            obs: Union["numpy.ndarray", "tensorflow.Tensor"]) -> int:
        """ Given an observation of the environment's current state, chooses an
        action based on the agent's policy.

        Args:
            obs (Union["numpy.ndarray", "tensorflow.Tensor"]): Observation of
                the environment's current state.

        Returns:
            An integer representing the chosen action (discrete).
        """

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        """ Triggers a new learning session, in which the agent will update its
        policy according to its past experience.
        """
