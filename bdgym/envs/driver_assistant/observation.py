"""Observation Classes for BDGym Highway Env """
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from gym import spaces

from highway_env.road.lane import AbstractLane
from highway_env.envs.common.observation import \
    KinematicObservation, ObservationType

import bdgym.envs.utils as utils
from bdgym.envs.driver_assistant.policy import GuidedIDMDriverPolicy
if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class DriverAssistantObservation(ObservationType):
    """Observation for the Driver Assistant Environment

    Handles observation and observation spaces for both Driver and
    assistant
    """

    NORM_OBS_LOW: float = -1.0
    NORM_OBS_HIGH: float = 1.0

    FEATURES = KinematicObservation.FEATURES

    DRIVER_FEATURES: List[str] = \
        KinematicObservation.FEATURES + ['acceleration', 'steering']
    DRIVER_EGO_ROW = 0

    ASSISTANT_FEATURES: List[str] = KinematicObservation.FEATURES
    ASSISTANT_EGO_ROW = 0

    def __init__(self, env: 'DriverAssistantEnv', obs_config: dict) -> None:
        self.env = env
        self.config = obs_config
        self.normalize = obs_config.get("normalize", True)
        self.clip = obs_config.get("clip", False)
        self.absolute = obs_config.get("absolute", False)
        self.features_range = obs_config.get("features_range", None)
        if self.features_range is None:
            self.get_features_range()

        self.vehicles_count: int = obs_config.get("vehicles_count", 5)
        self.assistant_type = KinematicObservation(
            env=env,
            features=self.ASSISTANT_FEATURES,
            vehicles_count=obs_config.get("vehicles_count", 5),
            features_range=obs_config.get("features_range", None),
            absolute=True,
            order=obs_config.get("order", "sorted"),
            normalize=False,
            clip=False,
            see_behind=obs_config.get("see_behind", False),
            observe_intentions=False
        )
        # These are the last unnormalized, absolute observations
        self.last_assistant_obs = np.zeros(self.assistant_space().shape)
        self.last_driver_obs = np.zeros(self.driver_space().shape)

    def space(self) -> spaces.Space:
        return self.assistant_space()

    def observe(self) -> np.ndarray:
        return self.observe_assistant()

    def driver_space(self) -> spaces.Space:
        """Get the driver's observation space """
        if self.normalize:
            low = np.zeros(self.driver_shape)
            high = np.zeros(self.driver_shape)
            for i, feature in enumerate(self.DRIVER_FEATURES):
                frange = self.features_range.get(
                    feature, [self.NORM_OBS_LOW, self.NORM_OBS_HIGH]
                )
                low[:, i] = frange[0]
                high[:, i] = frange[1]
            return spaces.Box(low=low, high=high, dtype=np.float32)

        return spaces.Box(
            shape=self.driver_shape,
            low=self.NORM_OBS_LOW,
            high=self.NORM_OBS_HIGH,
            dtype=np.float32
        )

    def assistant_space(self) -> spaces.Space:
        """Get the assistant's observation space """
        if self.normalize:
            low = np.zeros(self.assistant_shape)
            high = np.zeros(self.assistant_shape)
            for i, feature in enumerate(self.ASSISTANT_FEATURES):
                frange = self.features_range.get(
                    feature, [self.NORM_OBS_LOW, self.NORM_OBS_HIGH]
                )
                low[:, i] = frange[0]
                high[:, i] = frange[1]
            return spaces.Box(low=low, high=high, dtype=np.float32)

        return spaces.Box(
            shape=self.assistant_shape,
            low=self.NORM_OBS_LOW,
            high=self.NORM_OBS_HIGH,
            dtype=np.float32
        )

    def normalize_assistant_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize assistant observation

        Parameters
        ----------
        obs : np.ndarray
            the unnormalized assistant observation

        Returns
        -------
        no.ndarray
            assistant observation with values normalized to the range
            [DriverAssistantObservation.NORM_OBS_LOW,
             DriverAssistantObservation.NORM_OBS_HIGH].
        """
        return self._normalize_obs(obs, self.ASSISTANT_FEATURES)

    def normalize_driver_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize driver observation

        Parameters
        ----------
        obs : np.ndarray
            the unnormalized driver observation

        Returns
        -------
        no.ndarray
            driver observation with values normalized to the range
            [DriverAssistantObservation.NORM_OBS_LOW,
             DriverAssistantObservation.NORM_OBS_HIGH].
        """
        return self._normalize_obs(obs, self.DRIVER_FEATURES)

    def unnormalize_assistant_obs(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize assistant observation

        Parameters
        ----------
        obs : np.ndarray
            the normalized assistant observation with values in the range
            [DriverAssistantObservation.NORM_OBS_LOW,
             DriverAssistantObservation.NORM_OBS_HIGH]

        Returns
        -------
        no.ndarray
            assistant observation with unnormalized values
        """
        return self._unnormalize_obs(obs, self.ASSISTANT_FEATURES)

    def unnormalize_driver_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize driver observation

        Parameters
        ----------
        obs : np.ndarray
            the normalized driver observation with values in the range
            [DriverAssistantObservation.NORM_OBS_LOW,
             DriverAssistantObservation.NORM_OBS_HIGH]

        Returns
        -------
        no.ndarray
            driver observation with unnormalized values
        """
        return self._unnormalize_obs(obs, self.DRIVER_FEATURES)

    def _normalize_obs(self,
                       obs: np.ndarray,
                       features: List[str]) -> np.ndarray:
        features_range = self.get_features_range()

        norm_obs = np.zeros_like(obs)
        for feature, f_range in features_range.items():
            if feature not in features:
                continue
            f_idx = features.index(feature)
            norm_obs[:, f_idx] = utils.lmap_array(
                obs[:, f_idx],
                [f_range[0], f_range[1]],
                [self.NORM_OBS_LOW, self.NORM_OBS_HIGH]
            )
        if self.clip:
            np.clip(norm_obs, -1, 1, out=norm_obs)
        return norm_obs

    def _unnormalize_obs(self,
                         obs: np.ndarray,
                         features: List[str]) -> np.ndarray:
        features_range = self.get_features_range()

        norm_obs = np.zeros_like(obs)
        for feature, f_range in features_range.items():
            if feature not in features:
                continue
            f_idx = features.index(feature)
            norm_obs[:, f_idx] = utils.lmap_array(
                obs[:, f_idx],
                [self.NORM_OBS_LOW, self.NORM_OBS_HIGH],
                [f_range[0], f_range[1]]
            )
        if self.clip:
            np.clip(norm_obs, -1, 1, out=norm_obs)
        return norm_obs

    def observe_assistant(self,
                          normalize: Optional[bool] = None,
                          absolute: Optional[bool] = None) -> np.ndarray:
        """Get the assistant observation.

        Parameters
        ----------
        normalize : Optional[bool]
            whether the returned observation should be normalized or not. If
            None will default to the configured value stored in self.normalize
            (default=None).
        absolute : Optional[bool]
            whether the returned observation should contain absolute feature
            values (True) or relative values (False). If None will default to
            the configured value stored in self.absolute (default=None).

        Returns
        -------
        np.ndarray
            the assistant observation
        """
        obs = self.assistant_type.observe()

        self.last_assistant_obs = obs
        if absolute is False or (absolute is None and not self.absolute):
            obs = self.convert_to_relative_obs(obs, self.ASSISTANT_EGO_ROW)

        if normalize or (normalize is None and self.normalize):
            obs = self.normalize_assistant_obs(obs)
        return obs

    def observe_driver(self,
                       normalize: Optional[bool] = None,
                       absolute: Optional[bool] = None) -> np.ndarray:
        """Get the driver observation.

        Parameters
        ----------
        normalize : Optional[bool]
            whether the returned observation should be normalized or not. If
            None will default to the configured value stored in self.normalize
            (default=None).
        absolute : Optional[bool]
            whether the returned observation should contain absolute feature
            values (True) or relative values (False). If None will default to
            the configured value stored in self.absolute (default=None).

        Returns
        -------
        np.ndarray
            the driver observation
        """
        # This is the unnormalized action
        assistant_action = self.env.action_type.last_assistant_action
        obs = np.zeros(self.driver_shape, dtype=np.float32)
        presence_idx = self.DRIVER_FEATURES.index('presence')
        obs[self.DRIVER_EGO_ROW][presence_idx] = 1.0
        obs[self.DRIVER_EGO_ROW][presence_idx+1:] = assistant_action
        # Set other vehicle obs same as assistants obs
        # Ignoring last two columns, which are recommended action columns
        other_vehicle_obs = self.last_assistant_obs[self.ASSISTANT_EGO_ROW+1:]
        obs[1:, :-2] = other_vehicle_obs

        self.last_driver_obs = obs
        if absolute is False or (absolute is None and not self.absolute):
            obs = self.convert_to_relative_obs(obs, 0)

        if normalize or (normalize is None and self.normalize):
            obs = self.normalize_driver_obs(obs)

        return obs

    def convert_to_relative_obs(self,
                                obs: np.ndarray,
                                ego_vehicle_row: int = 0) -> np.ndarray:
        """Convert from absolute values to relative values in observation """
        origin = obs[ego_vehicle_row, :len(self.FEATURES)]
        rel_obs = np.array(obs)
        rel_obs[ego_vehicle_row+1:, :len(self.FEATURES)] = \
            obs[ego_vehicle_row+1:, :len(self.FEATURES)] - origin
        return rel_obs

    def convert_to_absolute_obs(self,
                                obs: np.ndarray,
                                ego_vehicle_row: int = 0) -> np.ndarray:
        """Convert from relative values to absolute values in observation """
        origin = obs[ego_vehicle_row, :len(self.FEATURES)]
        rel_obs = np.array(obs)
        rel_obs[ego_vehicle_row+1:, :len(self.FEATURES)] = \
            obs[ego_vehicle_row+1:, :len(self.FEATURES)] + origin
        return rel_obs

    @property
    def driver_shape(self) -> Tuple[int, int]:
        """The driver observation's shape """
        return (self.vehicles_count, len(self.DRIVER_FEATURES))

    @property
    def assistant_shape(self) -> Tuple[int, int]:
        """The Assistant observation's shape """
        return (self.vehicles_count, len(self.ASSISTANT_FEATURES))

    def get_features_range(self) -> dict:
        """Get the range of value observation features can take.

        Returns
        -------
        dict[str, List[float]]
           map from observation feature to the [min, max] values
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(
                self.env.vehicle.lane_index
            )
            max_speed = self.env.SPEED_UPPER_LIMIT
            max_acc = self.env.ACC_UPPER_LIMIT
            max_steering = GuidedIDMDriverPolicy.MAX_STEERING_ANGLE
            max_x = (
                self.env.config["duration"]
                / self.env.config["simulation_frequency"]
                * max_speed
            )
            max_y = AbstractLane.DEFAULT_WIDTH * len(side_lanes)
            self.features_range = {
                "x": [-max_x, max_x],
                "y": [-max_y, max_y],
                "vx": [-max_speed, max_speed],
                "vy": [-max_speed, max_speed],
                "acceleration": [-max_acc, max_acc],
                "steering": [-max_steering, max_steering]
            }
        return self.features_range


class DiscreteDriverAssistantObservation(DriverAssistantObservation):
    """Observation for the Discrete Driver Assistant Environment.

    This only alters the Assistant's observation which now includes the
    current offset being applied as the first row of the observation.
    """

    ASSISTANT_OFFSET_ROW = 0
    ASSISTANT_EGO_ROW = 1

    def observe_assistant(self,
                          normalize: Optional[bool] = None,
                          absolute: Optional[bool] = None) -> np.ndarray:
        """Get the observation for assistant """
        vehicle_obs = self.assistant_type.observe()

        assistant_action_type = self.env.action_type.assistant_action_type
        offset_obs = assistant_action_type.get_normalized_offset()

        obs = np.zeros(shape=self.assistant_shape, dtype=np.float32)
        # for offset obs ignore first column which is 'presence'
        obs[self.ASSISTANT_OFFSET_ROW, 1:] = offset_obs
        obs[self.ASSISTANT_EGO_ROW:] = vehicle_obs

        self.last_assistant_obs = obs
        if absolute is False or (absolute is None and not self.absolute):
            obs = self.convert_to_relative_obs(obs, self.ASSISTANT_EGO_ROW)

        if normalize or (normalize is None and self.normalize):
            return self.normalize_assistant_obs(obs)
        return obs

    @property
    def assistant_shape(self) -> Tuple[int, int]:
        """The Assistant observation's shape """
        return (self.vehicles_count+1, len(self.ASSISTANT_FEATURES))


def observation_factory(env: 'DriverAssistantEnv',
                        config: dict) -> DriverAssistantObservation:
    """Factory for observation type """
    if config["type"] == "DriverAssistantObservation":
        return DriverAssistantObservation(env, config)
    if config["type"] == "DiscreteDriverAssistantObservation":
        return DiscreteDriverAssistantObservation(env, config)
    raise ValueError(
        "Unsupported Observation Type for the DriverAssistant Env: "
        f"'{config['type']}."
    )
