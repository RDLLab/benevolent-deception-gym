"""Driver Assistant Env where driver policy is fixed.

It adds the guidance agent to the environment, when the driver
is a fixed policy.
"""
import time
from typing import Tuple

from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import Observation

import bdgym.envs.utils as utils
from bdgym.envs.driver_assistant.env import DriverAssistantEnv
from bdgym.envs.driver_assistant.policy import GuidedIDMDriverPolicy
from bdgym.envs.driver_assistant.action import DriverAssistantAction
from bdgym.envs.driver_assistant.observation import DriverAssistantObservation


class FixedDriverDriverAssistantEnv(DriverAssistantEnv):
    """Driver Assistant Env where driver policy is fixed. """

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        driver_config = self.config.get("driver_policy", {})
        self.driver_policy = GuidedIDMDriverPolicy.create_from(
            self.controlled_vehicles[0], **driver_config
        )

    def define_spaces(self) -> None:
        """Overrides Parent """
        self.observation_type = DriverAssistantObservation(
            self, **self.config["observation"]
        )
        self.action_type = DriverAssistantAction(self, self.config["action"])
        self.observation_space = self.observation_type.assistant_space()
        self.action_space = self.action_type.assistant_space()

    def default_config(self) -> dict:
        """Overrides Parent """
        config = super().default_config()
        config.update({
            "driver_policy": {
                "type": "GuidedIDMDriverVehicle",
                "independence": 0.75
            },
        })
        return config

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        action = self.action_type.get_assistant_absolute_action(action)
        self.action_type.assistant_act(action)

        driver_obs = self.observation_type.observe_driver(action)
        dt = 1 / self.config["simulation_frequency"]
        driver_action = self.driver_policy.get_action(driver_obs, dt)

        # Need to set this so Action type knows it's drivers turn to act
        self.next_agent = self.DRIVER_IDX
        self.steps += 1
        self._simulate(driver_action)
        obs = self.observation_type.observe_assistant()

        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        self._last_reward = reward
        self._last_action[self.ASSISTANT_IDX] = action
        self._last_action[self.DRIVER_IDX] = driver_action
        self.next_agent = self.ASSISTANT_IDX

        return obs, reward, terminal, info

    def _reset(self) -> None:
        super()._reset()
        driver_config = self.config.get("driver_policy", {})
        self.driver_policy = GuidedIDMDriverPolicy.create_from(
            self.controlled_vehicles[0], **driver_config
        )
