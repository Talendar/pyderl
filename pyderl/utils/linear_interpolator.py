"""
Based on: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/schedules.py#L76
"""


class LinearInterpolator:
    """ Handles the linear interpolation between some initial value and a final
     value over a pre-defined number of timesteps. A

     Args:
         initial_value (float): Initial value.
         final_value (float): Final value.
         num_timesteps (int): Total number of timesteps.
    """

    def __init__(self,
                 initial_value: float,
                 final_value: float,
                 num_timesteps: int) -> None:
        self._initial_value = initial_value
        self._final_value = final_value
        self._num_timesteps = num_timesteps

    def value(self, t):
        pc = min(t / self._num_timesteps, 1)
        var = self._final_value - self._initial_value
        return self._initial_value + pc * var

