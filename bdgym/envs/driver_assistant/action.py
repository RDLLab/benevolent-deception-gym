"""Action class for BDGym Highway Autopilot environment """
from typing import TYPE_CHECKING, Callable

import numpy as np
from gym import spaces

from highway_env import utils
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import \
    ContinuousAction, Action, action_factory, ActionType

from bdgym.envs.driver_assistant.policy import DriverAssistantVehicle

if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


# TODO
# 1. Implement Discrete Assistant Action Space
# 2. Fix up Assistant Continuous Offset Action Space


class DriverAssistantAction(ActionType):
    """Joint Driver and Assistant Action Type

    Essentially wraps:
    0. AssistantContinuousAction for the Assistant, and
    1. highway_env.envs.common.action.ContinousAction for the Driver
    """

    def __init__(self,
                 env: 'DriverAssistantEnv',
                 action_config: dict,
                 **config) -> None:
        super().__init__(env, **config)
        self.action_config = action_config
        self.driver_action_type = ContinuousAction(
            env, **action_config.get("driver", {})
        )
        self.assistant_action_type = assistant_action_factory(
            env, action_config["assistant"]
        )

    def space(self) -> spaces.Space:
        return spaces.Tuple([self.driver_space(), self.assistant_space()])

    @property
    def vehicle_class(self) -> Callable:
        return DriverAssistantVehicle

    @property
    def last_assistant_action(self) -> Action:
        """The last action performed by the Assistant """
        return self.assistant_action_type.last_action

    def driver_space(self) -> spaces.Space:
        """Get the driver action space """
        return self.driver_action_type.space()

    def assistant_space(self) -> spaces.Space:
        """Get the assistant action space """
        return self.assistant_action_type.space()

    def act(self, action: Action) -> None:
        if self.env.next_agent == self.env.ASSISTANT_IDX:
            return self.assistant_act(action)
        return self.driver_act(action)

    def driver_act(self, action: Action) -> None:
        """Execute driver action """
        self.driver_action_type.act(action)

    def assistant_act(self, action: Action) -> None:
        """Execute assistant action """
        self.assistant_action_type.act(action)

    def get_assistant_absolute_action(self, action: Action) -> Action:
        """Get action in terms of absolute distances/values """
        return self.assistant_action_type.get_absolute_action(action)


class AssistantContinuousAction(ContinuousAction):
    """Continuous action space for autopilot.

    This includes continuous actions for the autopilot signal sent
    to the driver which includes: ['x', 'y', 'vx', 'vy'] of the vehicle.

    It also includes autopilot recommendation for the drivers next action in
    terms of throttle and steering.

    The space intervals are always [-1, 1], but mapped to the proper values in
    the environment step function, as needed.
    """

    ASSISTANT_ACTION_SPACE_SIZE = 6
    ASSISTANT_ACTION_INDICES = {
        'x': 0,
        'y': 1,
        'vx': 2,
        'vy': 3,
        'acceleration': 4,
        'steering': 5
    }

    def __init__(self,
                 env: 'DriverAssistantEnv',
                 features_range: dict,
                 **config) -> None:
        super().__init__(env, **config)
        self.features_range = features_range

    def space(self) -> spaces.Box:
        """ Overrides ContinousAction.space() """
        shape = (self.ASSISTANT_ACTION_SPACE_SIZE,)
        return spaces.Box(-1., 1., shape=shape, dtype=np.float32)

    def act(self, action: Action) -> None:
        """ Overrides parent """
        self.last_action = action

    def get_absolute_action(self, action: Action) -> Action:
        """Get action in terms of absolute distances/values """
        return self.unnormalize_action(action)

    def normalize_action(self, action: Action) -> Action:
        """Convert action from absolute to proportional """
        norm_interval = [-1.0, 1.0]
        absolute_action = np.ones_like(action, dtype=np.float32)
        for i, frange in enumerate(self.features_range.values()):
            absolute_action[i] = utils.lmap(
                action[i], frange, norm_interval
            )
        return absolute_action

    def unnormalize_action(self, action: Action) -> Action:
        """Convert action from proportional to absolute """
        norm_interval = [-1.0, 1.0]
        absolute_action = np.ones_like(action, dtype=np.float32)
        for i, frange in enumerate(self.features_range.values()):
            absolute_action[i] = utils.lmap(
                action[i], norm_interval, frange
            )
        return absolute_action


class AssistantContinuousOffsetAction(AssistantContinuousAction):
    """Continuous action space for autopilot.

    Same as AutopilotContinuousAction except this time the action specifies
    how much to offset each signal from the value observed by the autopilot,
    and how much to offset from 0.0 for the acceleration and steering
    recommendation.

    For the recommendation this is essentially unchanged from
    AutopilotContinuousAction.
    """

    def get_last_ego_obs(self) -> Observation:
        """Get the last assistant observation for ego vehicle """
        last_obs = self.env.observation_type.get_last_assistant_frame()
        # include only first row which is the observation of controlled vehicle
        # also exclude first column, which indicated 'presence'
        return last_obs[0, 1:]

    def get_absolute_action(self, action: Action) -> Action:
        """Get action in terms of absolute distances/values """
        if action is None:
            action = self.last_action

        abs_offset_action = self.unnormalize_action(action)
        ego_obs = self.get_last_ego_obs()
        abs_signal = ego_obs + abs_offset_action[:4]
        abs_action = np.concatenate([abs_signal, abs_offset_action[4:]])

        self.last_action = action
        return abs_action


def assistant_action_factory(env: 'DriverAssistantEnv',
                             config: dict) -> ActionType:
    """Factory for action type """
    if config["type"] == "AssistantContinuousAction":
        return AssistantContinuousAction(env, **config)
    if config["type"] == "AssistantContinuousOffsetAction":
        return AssistantContinuousOffsetAction(env, **config)
    return action_factory(env, config)
