"""Observation Classes for BDGym Highway Env """
from typing import TYPE_CHECKING

import numpy as np
from gym import spaces

from highway_env.envs.common.observation import KinematicObservation

import bdgym.envs.utils as utils
if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class StackedKinematicObservation(KinematicObservation):
    """Observe kinematics of nearby vehicles.

    Supports frame stacking
    """

    OBS_LOW = -1.0
    OBS_HIGH = 1.0

    def __init__(self,
                 env: 'DriverAssistantEnv',
                 stack_size: int = 1,
                 **kwargs: dict) -> None:
        super().__init__(env=env, **kwargs)
        self.stack_size = stack_size
        if self.stack_size == 1:
            self.shape = (self.vehicles_count, len(self.features))
        else:
            self.shape = (
                self.vehicles_count, len(self.features), self.stack_size
            )
        self.state = np.zeros(self.shape)

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=self.shape,
            low=self.OBS_LOW,
            high=self.OBS_HIGH,
            dtype=np.float32
        )

    def observe(self) -> np.ndarray:
        new_obs = super().observe()
        if self.stack_size == 1:
            self.state = new_obs
            return new_obs
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state


class DriverAssistantObservation(StackedKinematicObservation):
    """Observation for the Driver Assistant Environment

    Handles observation and observation spaces for both Driver and
    assistant
    """

    def __init__(self, env: 'DriverAssistantEnv', **kwargs: dict) -> None:
        super().__init__(env=env, **kwargs)
        self.driver_state = np.zeros(self.driver_space().shape)

    def driver_space(self) -> spaces.Space:
        """Get the driver's observation space """
        shape = (self.vehicles_count, len(self.features)+2, self.stack_size)
        return spaces.Box(
            shape=shape,
            low=self.OBS_LOW,
            high=self.OBS_HIGH,
            dtype=np.float32
        )

    def assistant_space(self) -> spaces.Space:
        """Get the assistant's observation space """
        return super().space()

    def observe_assistant(self) -> np.ndarray:
        """Get the observation for assistant """
        return super().observe()

    def observe_driver(self, assistant_action: np.ndarray) -> np.ndarray:
        """Get the driver observation """
        latest_assistant_frame = self.get_last_assistant_frame()

        driver_frame = np.zeros(
            (self.vehicles_count, len(self.features)+2), dtype=np.float32
        )
        # 'presence'
        driver_frame[0][0] = 1.0
        # Set ego vehicle obs to assistant action
        driver_frame[0][1:] = assistant_action
        # Set other vehicle obs same as assistants obs
        # Ignoring last two columns
        driver_frame[1:, :-2] = latest_assistant_frame[1:]

        # print()
        # print("Observer_driver()")
        # print(
        #     "Latest Assistant Frame:\n",
        #     utils.np_array_str(latest_assistant_frame)
        # )
        # print("\nAssistant action:\n", utils.np_array_str(assistant_action))
        # print("\nDriver frame:\n", utils.np_array_str(driver_frame))
        # print()
        # input()

        if self.stack_size == 1:
            self.driver_state = driver_frame
            return driver_frame

        self.driver_state = np.roll(self.driver_state, -1, axis=-1)
        self.driver_state[:, :, -1] = driver_frame
        return self.driver_state

    def get_last_assistant_frame(self) -> np.ndarray:
        """Get the last frame observed by assistant """
        if self.stack_size == 1:
            return self.state
        return self.state[:, :, -1]
