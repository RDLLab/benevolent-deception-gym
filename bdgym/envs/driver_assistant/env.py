"""Driver Assistant Env.

Credit to the HighwayEnv Gym environment which this environment is built on top
of: https://github.com/eleurent/highway-env
"""
from typing import Optional, Tuple

import numpy as np

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.abstract import Observation
from highway_env.vehicle.controller import ControlledVehicle

from bdgym.envs.driver_assistant.action import DriverAssistantAction
from bdgym.envs.driver_assistant.policy import GuidedIDMDriverPolicy
from bdgym.envs.driver_assistant.observation import DriverAssistantObservation
from bdgym.envs.driver_assistant.graphics import \
    DriverAssistantEnvViewer, AssistantActionDisplayer


class DriverAssistantEnv(HighwayEnv):
    """Driver Assistant Environment.

    Builds on the HighwayEnv Gym environment:
        https://github.com/eleurent/highway-env

    Description:
        The HighwayEnv involves a driver agent controlling a vehicle as it
        navigates a highway containing other vehicles. The driver's goal is to
        driver along the highway while avoiding collisions, with some
        incentives to be in the overtaking (right) lane and going fast.

        The DriverAssistantEnv adds a Driver Assistant into the environment,
        who has the same goal as the driver. The Driver Assistant is able to
        modify the observations that the driver observes about the controlled
        vehicle. Specifically, it can modify the observation of the current
        position of the vehicle 'x', 'y' and the current velocity 'vx', 'vy'.
        Additionally, the Driver Assistant supplies recommended actions to the
        driver 'acceleration', 'steering', which the driver observes and can
        act on as desired.

    State & Starting State:
        The state is the same as for the HighwayEnv.

    Reward:
        The reward is similarly as for the HighwayEnv, except the insentives
        for being in the right lane and going at a high speed are increased.

    Environment Interaction:
        The driver and assistant takes turns in performing action. Firstly,
        the assistant recieves the observation from the environment and
        performs their action (i.e. observation modification and recommended
        action). The result of this action is then the observation for the
        driver, which then performs their driving action (i.e. acceleration and
        steering), which affects the car in the environment. The observation
        following the drivers action is then the observation for the assistant.

        Note that the external environment is only affected when the driver
        performs their action.

    Driver Properties
    -----------------
    Observation:
        Type: Box(V+1, 7)
        Is the same as for HighwayEnv, where V is number of nearby vehicles
        observed. The first 5 of 7 column are: 'presence', 'x', 'y', 'vx',
        'vy' of the ego vehicle and the V nearby vehicles. The last two columns
        are the recommended actions from the driver assistant: 'acceleration'
        and 'steering'. The first row is always the ego vehicles observation.
        For all rows except the ego vehicle row (first) the last two columns
        are 0.0. For the driver 'presence' is always 1.0.

        The key difference from the HighwayEnv is the addition of the driver
        assistant recommended actions and also that the observation of the
        ego vehicle is controlled by the driver assistant (except
        for 'presence' which is always 1.0 for the ego vehicle).

    Actions:
        Type: Box(2)
        These are unchanged from the HighwayEnv:
        Num   Observation                           Min       Max
        0     acceleration                          -1.0      1.0
        1     steering                              -1.0      1.0

    Driver Assistant (AKA Assistant) Properties
    -------------------------------------------
    Observation:
        Type: Box(V, 5)
        The Assistant observation is the exact observation from the orignal
        HighwayEnv: 'presence', 'x', 'y', 'vx', 'vy' of the ego vehicle and the
        V nearby vehicles.

    Actions:
        Type: Box(6)
        These are unchanged from the HighwayEnv:
        Num   Observation                           Min       Max
        0     x                                     -1.0      1.0
        1     y                                     -1.0      1.0
        2     vx                                    -1.0      1.0
        3     vy                                    -1.0      1.0
        4     acceleration                          -1.0      1.0
        5     steering                              -1.0      1.0

    """

    RIGHT_LANE_REWARD: float = 0.2
    HIGH_SPEED_REWARD: float = 0.8

    SPEED_UPPER_LIMIT = GuidedIDMDriverPolicy.MAX_SPEED
    ACC_UPPER_LIMIT = 10.0

    OTHER_VEHICLE_OBS_NOISE = 0.0    # [m]

    ASSISTANT_IDX = 0
    """Index of assistant agent (i.e. in turn and action and obs spaces) """

    DRIVER_IDX = 1
    """Index of driver agent (i.e. in turn and action and obs spaces) """

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.next_agent = self.ASSISTANT_IDX
        self.agent_display = None
        self._last_reward = 0.0
        self._last_action = [
            np.zeros(self.action_type.assistant_space().shape),
            np.zeros(self.action_type.driver_space().shape)
        ]

    def define_spaces(self) -> None:
        """Overrides Parent """
        self.observation_type = DriverAssistantObservation(
            self, **self.config["observation"]
        )
        self.action_type = DriverAssistantAction(self, self.config["action"])
        self.observation_space = [
            self.observation_type.assistant_space(),
            self.observation_type.driver_space()
        ]
        self.action_space = [
            self.action_type.assistant_space(),
            self.action_type.driver_space()
        ]

    def default_config(self) -> dict:
        """Overrides Parent """
        config = super().default_config()

        duration = config["duration"] * config["simulation_frequency"]
        max_speed = self.SPEED_UPPER_LIMIT
        max_acc = self.ACC_UPPER_LIMIT
        max_steering = GuidedIDMDriverPolicy.MAX_STEERING_ANGLE
        max_x = config["duration"] * max_speed
        max_y = 4 * AbstractLane.DEFAULT_WIDTH
        config.update({
            "observation": {
                "type": "DriverAssistantObservation",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                        "x": [-max_x, max_x],
                        "y": [-max_y, max_y],
                        "vx": [-max_speed, max_speed],
                        "vy": [-max_speed, max_speed]
                },
                "normalize": True,
                "absolute": False,
                "order": "sorted",
                "stack_size": 1
            },
            "action": {
                "assistant": {
                    "type": "AssistantContinuousAction",
                    "features_range": {
                        "x": [-max_x, max_x],
                        "y": [-max_y, max_y],
                        "vx": [-max_speed, max_speed],
                        "vy": [-max_speed, max_speed],
                        "acceleration": [-max_acc, max_acc],
                        "steering": [-max_steering, max_steering]
                    }
                },
                "driver": {
                    "type": "ContinuousAction",
                    "features_range": {
                        "acceleration": [-max_acc, max_acc],
                        "steering": [-max_steering, max_steering]
                    },
                    "acceleration_range": (-max_acc, max_acc),
                    "steering_range": (-max_steering, max_steering)
                }
            },
            "policy_frequency": config["simulation_frequency"],
            "duration": duration,
            "vehicles_density": 1,
            "vehicles_count": 25,    # Default = 50
            "reward_speed_range": [0, max_speed],
            "collision_reward": -1,
            "offroad_terminal": True,
            "action_display": True,
            "screen_width": 800,  # [px]
            "screen_height": 250,  # [px]
        })
        return config

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.next_agent == self.ASSISTANT_IDX:
            action = self.action_type.get_assistant_absolute_action(action)
            obs = self.observation_type.observe_driver(action)
        else:
            self.steps += 1
            self._simulate(action)
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
        self._last_action[self.next_agent] = action
        self.next_agent = (self.next_agent + 1) % (self.ASSISTANT_IDX + 1)

        return obs, reward, terminal, info

    def reset(self) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the initial assistant observation of the reset state
        """
        super().reset()
        self.next_agent = self.ASSISTANT_IDX
        self._last_reward = 0.0
        self._last_action = [
            np.zeros(self.action_type.assistant_space().shape),
            np.zeros(self.action_type.driver_space().shape)
        ]
        return self.observation_type.observe()

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Overrides parent """
        if self.viewer is None:
            self.viewer = DriverAssistantEnvViewer(self)
            if self.config["manual_control"] or self.config["action_display"]:
                self.agent_display = AssistantActionDisplayer(self)
                self.viewer.set_agent_display(self.agent_display)

        return super().render(mode)

    def _simulate(self, action: Optional[Action] = None) -> None:
        sim_freq = self.config["simulation_frequency"]
        pol_freq = self.config["policy_frequency"]
        for _ in range(int(sim_freq // pol_freq)):
            # Forward action to the vehicle
            if self.config["manual_control"]:
                self.action_type.act(None)
            elif (
                    action is not None
                    and self.time % int(sim_freq // pol_freq) == 0
            ):
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / sim_freq)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer
            # has been launched. Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def _reward(self, action: Action) -> float:
        if self.next_agent == self.ASSISTANT_IDX:
            return self._last_reward

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        if isinstance(self.vehicle, ControlledVehicle):
            lane = self.vehicle.target_lane_index[2]
        else:
            lane = self.vehicle.lane_index[2]

        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(
            reward,
            [
                self.config["collision_reward"],
                self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD
            ],
            [-1, 1]
        )
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    @property
    def last_driver_action(self) -> Action:
        """The last action performed by the driver agent """
        return self._last_action[self.DRIVER_IDX]

    @property
    def last_assistant_action(self) -> Action:
        """The last action performed by the assistant agent """
        return self._last_action[self.ASSISTANT_IDX]

    @property
    def last_assistant_obs(self) -> np.ndarray:
        """The last action performed by the assistant agent """
        return self.observation_type.get_last_assistant_frame()

    @property
    def delta_time(self) -> float:
        """The environment time step size """
        return 1 / self.config["simulation_frequency"]
