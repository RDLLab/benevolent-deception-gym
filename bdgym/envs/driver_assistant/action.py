"""Action class for BDGym Highway Autopilot environment """
from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np
from gym import spaces

from highway_env import utils
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import \
    ContinuousAction, Action, action_factory, ActionType


from bdgym.envs.driver_assistant.policy import DriverAssistantVehicle

if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class DriverAssistantAction(ActionType):
    """Joint Driver and Assistant Action Type

    Essentially wraps:
    0. AssistantContinuousAction, AssistantContinuousOffsetAction or
       AssistantDiscreteActionSpace for the Assistant, and
    1. highway_env.envs.common.action.ContinousAction for the Driver
    """

    def __init__(self,
                 env: 'DriverAssistantEnv',
                 action_config: dict) -> None:
        self.env = env
        self.action_config = action_config
        self.driver_action_type = ContinuousAction(
            env, **action_config.get("driver", {})
        )
        self.assistant_action_type = assistant_action_factory(
            env, action_config["assistant"]
        )

    def space(self) -> spaces.Space:
        return spaces.Tuple([self.driver_space(), self.assistant_space()])

    def reset(self):
        """Reset the action space following an episode """
        self.assistant_action_type.reset()
        # no need to reset driver type since it is the ContinousAction type

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


class AssistantContinuousAction(ContinuousAction):
    """Continuous action space for assistant.

    This includes continuous actions for the assistant signal sent
    to the driver which includes: ['x', 'y', 'vx', 'vy'] of the vehicle.

    It also includes assistant recommendation for the drivers next action in
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

    def reset(self):
        """Reset the action space following an episode """
        self.last_action = np.zeros(
            self.ASSISTANT_ACTION_SPACE_SIZE, dtype=np.float32
        )

    def act(self, action: Action) -> None:
        """ Overrides parent """
        self.last_action = self.unnormalize_action(action)

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
    """Continuous action space for Assistant.

    Same as AssistantContinuousAction except this time the action specifies
    how much to offset each signal from the value observed by the assistant,
    and how much to offset from 0.0 for the acceleration and steering
    recommendation.

    For the recommendation this is essentially unchanged from
    AssistantContinuousAction.
    """

    def get_last_ego_obs(self) -> Observation:
        """Get the last assistant observation for ego vehicle """
        last_obs = self.env.observation_type.get_last_assistant_frame()
        # include only first row which is the observation of controlled vehicle
        # also exclude first column, which indicated 'presence'
        return last_obs[0, 1:]

    def act(self, action: Action) -> None:
        """ Overrides parent """
        abs_offset_action = self.unnormalize_action(action)
        ego_obs = self.get_last_ego_obs()
        abs_signal = ego_obs + abs_offset_action[:4]
        abs_action = np.concatenate([abs_signal, abs_offset_action[4:]])
        self.last_action = abs_action


class AssistantDiscreteActionSpace(ActionType):
    """Discrete Action space for Assistant.

    This is a MultiDiscrete Action space, where each action is a combination
    of 6 sub actions, similar to the continuous Assistant actions:

    Type: MultiDiscrete([3, 3, 3, 3, 3, 3])
    Num   Action Space    Actions
    0     x               NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2
    1     y               NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2
    2     vx              NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2
    3     vy              NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2
    4     acceleration    NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2
    5     steering        NOOP[0], UP[1], DOWN[2] - params: min: 0, max: 2

    For the ['x', 'y', 'vx', 'vy'] action spaces the actions have the
    effect of shifting the current offset/distortion being applied to the
    by a fixed amount (controlled by AssistantDiscreteActionSpace.STEP_SIZE
    parameter, default is 0.1 of max range of feature).

    The 'acceleration' and 'steering' action space actions have the effect
    of recommending to the driver to steer and/or accelerate up, down, or
    no change for the step the action is applied.

    NOTE: Using this action space affects the observation space of the
          assistant. Specifically, it adds an extra row at the top of the
          observation matrix which is the current offset.
    """

    STEP_SIZE_MAP = {
        'x': 0.001,
        'y': 0.05,
        'vx': 0.05,
        'vy': 0.05,
        'acceleration': 0.05,
        'steering': 0.025
    }
    """This step size of each action

    This is the proportion of the max range of a variable, so the actual
    step size will vary depending on the variable being affected.

    The proportion is set per variable since the range of each variable
    can be significantly different.
    """

    ASSISTANT_DISCRETE_ACTION_SPACE_SIZE = 6
    ASSISTANT_DISCRETE_ACTION_SPACE_SHAPE = [3, 3, 3, 3, 3, 3]
    ASSISTANT_DISCRETE_ACTION_INDICES = {
        'x': 0,
        'y': 1,
        'vx': 2,
        'vy': 3,
        'acceleration': 4,
        'steering': 5
    }

    NOOP = 0
    UP = 1
    DOWN = 2
    """Integer values of each discrete action """

    OFFSET_FEATURES = ['x', 'y', 'vx', 'vy']
    """List of feature that offset is applied too """

    def __init__(self,
                 env: 'DriverAssistantEnv',
                 features_range: dict,
                 **kwargs) -> None:
        self.env = env
        self.features_range = features_range
        # This is the last action as unnormalized, continuous assistant action
        self.last_action = np.full(
            self.ASSISTANT_DISCRETE_ACTION_SPACE_SIZE,
            self.NOOP,
            dtype=np.float32
        )
        # The current unnormalized offset
        self.current_offset = np.zeros(len(self.OFFSET_FEATURES))

        self.feature_step_size = {}
        for feature in self.ASSISTANT_DISCRETE_ACTION_INDICES:
            feat_range = self.features_range[feature]
            max_range = feat_range[1] - feat_range[0]
            step_proportion = self.STEP_SIZE_MAP[feature]
            self.feature_step_size[feature] = step_proportion*max_range

    def space(self) -> spaces.MultiDiscrete:
        """ Overrides ContinousAction.space() """
        return spaces.MultiDiscrete(self.ASSISTANT_DISCRETE_ACTION_SPACE_SHAPE)

    def reset(self):
        """Reset the action space following an episode """
        self.last_action = np.full(
            self.ASSISTANT_DISCRETE_ACTION_SPACE_SIZE,
            self.NOOP,
            dtype=np.float32
        )
        self.current_offset = np.zeros(len(self.OFFSET_FEATURES))

    def act(self, action: Action) -> None:
        """ Overrides parent

        Assumes action is normalized
        """
        if action is not None:
            # print("\nAction:", action)
            # print("init current_offset:", self.current_offset)
            self._update_current_offset(action)
            recommendation = self._get_recommendations(action)
        else:
            recommendation = self.last_action[len(self.OFFSET_FEATURES):]

        last_obs = self.get_last_ego_obs()

        abs_action = np.zeros(len(self.ASSISTANT_DISCRETE_ACTION_INDICES))
        # print("init abs_action:", abs_action)
        abs_action[:len(self.OFFSET_FEATURES)] = self.current_offset + last_obs
        # print("abs_action after obs:", abs_action)
        abs_action[len(self.OFFSET_FEATURES):] = recommendation
        # print("abs_action after rcmd:", abs_action)
        # print("Performing action:", abs_action)
        self.last_action = abs_action

    def _update_current_offset(self, action: Action) -> None:
        for feature in self.OFFSET_FEATURES:
            f_idx = self.ASSISTANT_DISCRETE_ACTION_INDICES[feature]
            f_action = action[f_idx]
            if f_action == self.UP:
                delta = self.feature_step_size[feature]
            elif f_action == self.DOWN:
                delta = -1 * self.feature_step_size[feature]
            else:
                delta = 0
            self.current_offset[f_idx] += delta
            self.current_offset[f_idx] = np.clip(
                self.current_offset[f_idx],
                self.features_range[feature][0],
                self.features_range[feature][1]
            )

    def _get_recommendations(self, action: Action) -> Tuple[float, float]:
        controls = []
        for feature in ['acceleration', 'steering']:
            f_idx = self.ASSISTANT_DISCRETE_ACTION_INDICES[feature]
            f_action = action[f_idx]
            if f_action == self.UP:
                delta = self.feature_step_size[feature]
            elif f_action == self.DOWN:
                delta = -1 * self.feature_step_size[feature]
            else:
                delta = 0
            controls.append(delta)
        return controls[0], controls[1]

    @property
    def vehicle_class(self) -> Callable:
        return DriverAssistantVehicle

    def get_last_ego_obs(self) -> Observation:
        """Get the last assistant observation for ego vehicle """
        last_obs = self.env.observation_type.get_last_assistant_frame()
        # include only first row which is the observation of controlled vehicle
        # also exclude first column, which indicated 'presence'
        return last_obs[0, 1:]

    def get_normalized_offset(self) -> np.ndarray:
        """Get the current offset in normalized form (values in [-1.0, 1.0])

        Returns
        -------
        np.ndarray
            Current offset with normalized values
        """
        norm_interval = [-1.0, 1.0]
        norm_offset = np.zeros(len(self.OFFSET_FEATURES))
        for feature in self.OFFSET_FEATURES:
            f_idx = self.ASSISTANT_DISCRETE_ACTION_INDICES[feature]
            f_range = self.features_range[feature]
            norm_offset[f_idx] = utils.lmap(
                self.current_offset[f_idx], f_range, norm_interval
            )
        return norm_offset


def assistant_action_factory(env: 'DriverAssistantEnv',
                             config: dict) -> ActionType:
    """Factory for action type """
    if config["type"] == "AssistantContinuousAction":
        return AssistantContinuousAction(env, **config)
    if config["type"] == "AssistantContinuousOffsetAction":
        return AssistantContinuousOffsetAction(env, **config)
    if config["type"] == "AssistantDiscreteActionSpace":
        return AssistantDiscreteActionSpace(env, **config)
    return action_factory(env, config)
