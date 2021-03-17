"""Driver Policies for the Driver Assistant Environment """
from copy import deepcopy
from typing import Tuple, Optional, List, Dict

import numpy as np

from highway_env.utils import not_zero, do_every
from highway_env.types import Vector
from highway_env.road.lane import AbstractLane
from highway_env.envs.common.action import Action
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, LaneIndex, Route

import bdgym.envs.utils as utils
from bdgym.envs.driver_assistant.driver_types import sample_driver_config

Observation = np.ndarray


class DriverAssistantVehicle(Vehicle):
    """A faster vehicle """

    MAX_SPEED = 50.


class IDMDriverPolicy(IDMVehicle):
    """A driver Policy that acts similar to IDMVehicle.

    Key difference is that it's decisions are based on the observations
    of it's own position and velocity that it recieves from the assistant and
    the noisy observations it recieves about the other nearby vehicles
    """

    MAX_SPEED = DriverAssistantVehicle.MAX_SPEED

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 acc_max: float = None,
                 comfort_acc_max: float = None,
                 comfort_acc_min: float = None,
                 distance_wanted: float = None,
                 time_wanted: float = None,
                 delta: float = None,
                 politeness: float = None,
                 lane_change_min_acc_gain: float = None,
                 lane_change_max_braking_imposed: float = None,
                 lane_change_delay: float = None,
                 **kwargs):
        super().__init__(
            road,
            position,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
            enable_lane_change=enable_lane_change,
            timer=timer
        )

        self.acc_max = self.ACC_MAX if acc_max is None else acc_max
        self.comfort_acc_max = self.COMFORT_ACC_MAX \
            if comfort_acc_max is None else comfort_acc_max
        self.comfort_acc_min = self.COMFORT_ACC_MIN \
            if comfort_acc_min is None else comfort_acc_min
        self.distance_wanted = self.DISTANCE_WANTED \
            if distance_wanted is None else distance_wanted
        self.time_wanted = self.TIME_WANTED \
            if time_wanted is None else time_wanted
        self.delta = self.DELTA if delta is None else delta
        self.politeness = self.POLITENESS if politeness is None else politeness
        self.lane_change_min_acc_gain = self.LANE_CHANGE_MIN_ACC_GAIN \
            if lane_change_min_acc_gain is None else lane_change_min_acc_gain
        self.lane_change_max_braking_imposed = \
            self.LANE_CHANGE_MAX_BRAKING_IMPOSED \
            if lane_change_max_braking_imposed is None \
            else lane_change_max_braking_imposed
        self.lane_change_delay = self.LANE_CHANGE_DELAY \
            if lane_change_delay is None else lane_change_delay

        if timer is None:
            self.timer = (np.sum(self.position)*np.pi) % self.lane_change_delay

    @classmethod
    def create_from(cls, vehicle: Vehicle, **kwargs) -> "IDMDriverPolicy":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        return cls(
            road=vehicle.road,
            position=vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            **kwargs
        )

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """
        Get the next action driver will take

        Note: assistant signal and other vehicle observations should
        be non-normalized values.

        :param obs: the driver observation of ego and neighbouring vehicles
            ['presence', 'x', 'y', 'vx', 'vy', 'acceleration', 'steering']
        :param dt: the step size for action
        :return: action ['acceleration', 'steering'] the vehicle would take
        """
        assistant_signal, other_vehicle_obs = self.parse_obs(obs)
        self._update_dynamics(assistant_signal, dt)
        return self._get_idm_action(other_vehicle_obs)

    @staticmethod
    def parse_obs(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Parse driver observation into ego and other vehicle obs """
        # for ego vehicle ignore 'presence' feature
        assistant_signal = obs[0][1:]
        # for other vehicle obs ignore 'acceleration' and 'steering' as these
        # are always 0.0 for other vehicles
        other_vehicle_obs = obs[1:, :-2]
        return assistant_signal, other_vehicle_obs

    def _update_dynamics(self, ego_vehicle_obs: Action, dt: float):
        self.position = ego_vehicle_obs[0:2]
        vx, vy = ego_vehicle_obs[2], ego_vehicle_obs[3]
        self.speed, self.heading = self._get_speed_and_heading(vx, vy)
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position
            )
            self.lane = self.road.network.get_lane(self.lane_index)
            # self.road.update_nearest_neighbours()

        self.timer += dt

    @staticmethod
    def _get_speed_and_heading(vx: float, vy: float) -> Tuple[float, float]:
        speed = np.sqrt(vx**2 + vy**2)
        if speed == 0.0:
            # vx = vy = 0.0
            heading = 0.0
        elif vx == 0.0:
            heading = np.arcsin(vy / speed)
        elif vy == 0.0:
            heading = np.arccos(vx / speed)
        else:
            heading = np.arctan(vy / vx)
        return speed, heading

    @staticmethod
    def _get_direction(heading: float) -> np.ndarray:
        return np.array([np.cos(heading), np.sin(heading)])

    @property
    def observation(self) -> np.ndarray:
        """Vehicle position and velocity in standard observation format """
        return np.array([1.0, *self.position, *self.velocity])

    def _get_idm_action(self, other_vehicle_obs: np.ndarray) -> np.ndarray:
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self._change_lane_policy(other_vehicle_obs)
        steering = self.steering_control(self.target_lane_index)
        steering = np.clip(
            steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle_obs, _ = self._get_neighbours(other_vehicle_obs)
        acceleration = self._acceleration_from_obs(front_vehicle_obs)
        acceleration = np.clip(acceleration, -1*self.acc_max, self.acc_max)
        return np.array([acceleration, steering])

    def _acceleration_from_obs(self, front_vehicle_obs: np.ndarray) -> float:
        acceleration = self.comfort_acc_max * (
            1 - np.power(max(self.speed, 0) / self.target_speed, self.delta)
        )

        if front_vehicle_obs is not None:
            front_pos = front_vehicle_obs[1:3]
            d = (
                self.lane.local_coordinates(front_pos)[0]
                - self.lane.local_coordinates(self.position)[0]
            )
            gap = self._desired_gap_from_obs(front_vehicle_obs)
            gap /= not_zero(d)
            acceleration -= self.comfort_acc_max * np.power(gap, 2)

        return acceleration

    def _acceleration(self,
                      ego_xy: Tuple[float, float],
                      ego_velocity: Tuple[float, float],
                      front_xy: Tuple[float, float] = None,
                      front_velocity: Tuple[float, float] = None,
                      lane: AbstractLane = None) -> float:
        ego_target_speed = not_zero(0)
        ego_speed, ego_heading = self._get_speed_and_heading(*ego_velocity)
        speed = max(ego_speed, 0) / ego_target_speed

        acceleration = self.comfort_acc_max * (
            1 - np.power(speed, self.delta)
        )

        if front_xy is not None:
            if lane is None:
                lane = self.lane

            d = (
                lane.local_coordinates(front_xy)[0]
                - lane.local_coordinates(ego_xy)[0]
            )
            ego_direction = self._get_direction(ego_heading)
            gap = self._desired_gap(
                ego_velocity, ego_speed, ego_direction, front_velocity
            )
            gap /= not_zero(d)
            acceleration -= self.comfort_acc_max * np.power(gap, 2)
        return acceleration

    def _desired_gap_from_obs(self, vehicle_obs: np.ndarray):
        other_velocity = vehicle_obs[3:4]
        return self._desired_gap(
            self.velocity, self.speed, self.direction, other_velocity
        )

    def _desired_gap(self,
                     ego_velocity: np.ndarray,
                     ego_speed: float,
                     ego_direction: float,
                     other_velocity: np.ndarray) -> float:
        d0 = self.distance_wanted
        tau = self.time_wanted
        ab = -1*self.comfort_acc_max * self.comfort_acc_min
        dv = np.dot(ego_velocity - other_velocity, ego_direction)
        d_star = d0 + ego_speed * tau + ego_speed * dv / (2 * np.sqrt(ab))
        return d_star

    def _get_neighbours(self,
                        other_vehicle_obs: np.ndarray,
                        lane_index: LaneIndex = None
                        ) -> Tuple[Optional[Vehicle], Optional[Vehicle]]:
        """Get closest vehicles to ego vehicle within specified lane """
        if lane_index is None:
            lane_index = self.lane_index

        lane = self.road.network.get_lane(lane_index)
        ego_lane_x = lane.local_coordinates(self.position)[0]

        closest_front = None
        min_front_distance = float('inf')
        closest_rear = None
        min_rear_distance = float('inf')

        for other in other_vehicle_obs:
            other_xy = other[1:3]
            if not lane.on_lane(other_xy):
                continue

            other_lane_x = lane.local_coordinates(other_xy)[0]
            distance = abs(other_lane_x - ego_lane_x)
            if ego_lane_x < other_lane_x and distance < min_front_distance:
                closest_front = other
                min_front_distance = distance
            elif ego_lane_x > other_lane_x and distance < min_rear_distance:
                closest_rear = other
                min_rear_distance = distance

        return closest_front, closest_rear

    def _change_lane_policy(self, other_vehicle_obs: np.ndarray) -> None:
        # at a given frequency,
        if not do_every(self.lane_change_delay, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                    self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self._mobil(lane_index, other_vehicle_obs):
                self.target_lane_index = lane_index

    def _mobil(self,
               lane_index: LaneIndex,
               other_vehicle_obs: np.ndarray) -> bool:
        """ Overrides parent """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self._get_neighbours(
            other_vehicle_obs, lane_index
        )

        lane = self.road.network.get_lane(lane_index)
        ego_obs = self.observation

        new_following_a, new_following_pred_a = self._acceleration_changes(
            following=new_following,
            preceding=new_preceding,
            new_preceding=ego_obs,
            lane=lane
        )
        if new_following_pred_a < -1*self.lane_change_max_braking_imposed:
            return False

        # Do I have a planned route for a specific lane which is safe for me
        # to access?
        old_preceding, old_following = self._get_neighbours(other_vehicle_obs)
        self_pred_a = self._acceleration_from_obs(new_preceding)

        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) \
               != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            if self_pred_a < -1*self.lane_change_max_braking_imposed:
                return False
        else:
            # Is there an acceleration advantage for me and/or my
            # followers to change lane?
            self_a = self._acceleration_from_obs(old_preceding)

            old_following_a, old_following_pred_a = self._acceleration_changes(
                following=old_following,
                preceding=ego_obs,
                new_preceding=old_preceding
            )

            jerk = self_pred_a - self_a
            jerk += self.politeness * (
                new_following_pred_a - new_following_a
                + old_following_pred_a - old_following_a
            )
            if jerk < self.lane_change_min_acc_gain:
                return False

        # All clear, let's go!
        return True

    def _acceleration_changes(self,
                              following: np.ndarray,
                              preceding: np.ndarray,
                              new_preceding: np.ndarray,
                              lane: AbstractLane = None
                              ) -> Tuple[float, float]:
        following_a = 0.0
        following_pred_a = 0.0
        if following is not None and preceding is not None:
            following_a = self._acceleration(
                ego_xy=following[1:3],
                ego_velocity=following[3:5],
                front_xy=preceding[1:3],
                front_velocity=preceding[3:5],
                lane=lane
            )

        if following is not None and new_preceding is not None:
            following_pred_a = self._acceleration(
                ego_xy=following[1:3],
                ego_velocity=following[3:5],
                front_xy=new_preceding[1:3],
                front_velocity=new_preceding[3:5],
                lane=lane
            )
        return following_a, following_pred_a


class IDMAssistantPolicy(IDMDriverPolicy):
    """An IDM Assistant policy that provides vehicle obs and recommendation

    Specifically:
    - Returns the actual observations recieved for the driver vehicle
    - Recommends acceleration and steering controls to the driver based on IDM
      model
    """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 *args,
                 action_ranges: Optional[Dict[str, List[float]]] = None,
                 normalize: bool = True,
                 **kwargs):
        super().__init__(road, position, *args, **kwargs)
        self.action_ranges = {}
        if action_ranges is not None:
            self.action_ranges = action_ranges
        self.normalize = normalize

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """
        Get the next action assistant will take

        :param full_obs: full observation of vehicle including of nearby
            vehicles. Includes ['presence', 'x', 'y', 'vx', 'vy']
        :return: the assistant action
            ['x', 'y', 'vx', 'vy', 'acceleration', 'steering']
        """
        other_vehicle_obs = obs[1:, :]
        assistant_signal = obs[0, 1:]
        self._update_dynamics(assistant_signal, dt)
        recommended_controls = self._get_idm_action(other_vehicle_obs)

        action = np.concatenate((assistant_signal, recommended_controls))

        if self.normalize:
            for i, frange in enumerate(self.action_ranges.values()):
                action[i] = utils.lmap(
                    action[i], frange, [-1, 1]
                )

        return action


class RandomAssistantPolicy(IDMAssistantPolicy):
    """A Random Assistant policy

    Specifically:
    - Returns the actual observations recieved for the driver vehicle
    - Recommends a random acceleration and steering control to the driver
    """

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """ Overrides IDMDriverVehicle.get_action() """
        action = super().get_action(obs, dt)
        action[4] = utils.get_truncated_normal(0.0, 1.0, -1.0, 1.0)
        action[5] = utils.get_truncated_normal(0.0, 1.0, -1.0, 1.0)
        return action


class RandomDiscreteAssistantPolicy(IDMAssistantPolicy):
    """A Random Assistant policy for Discrete Assistant Action

    Specifically:
    - Returns the actual observations recieved for the driver vehicle
    - Recommends a random acceleration and steering control to the driver
    """
    DISCRETE_ACTION_SPACE_SIZE = 6

    NOOP = 0
    UP = 1
    DOWN = 2
    """Integer values of each discrete action """

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """ Overrides IDMDriverVehicle.get_action() """
        action = np.full(self.DISCRETE_ACTION_SPACE_SIZE, self.NOOP)
        action[4] = np.random.choice([self.NOOP, self.UP, self.DOWN])
        action[5] = np.random.choice([self.NOOP, self.UP, self.DOWN])
        return action


class RandomDriverPolicy(IDMDriverPolicy):
    """A Random driver policy """

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """ Overrides IDMDriverVehicle.get_action() """
        assistant_signal, _ = self.parse_obs(obs)
        self._update_dynamics(assistant_signal, dt)
        return np.random.uniform(
            low=[-1*self.acc_max, -1*self.MAX_STEERING_ANGLE],
            high=[self.acc_max, self.MAX_STEERING_ANGLE],
        )


class GuidedIDMDriverPolicy(IDMDriverPolicy):
    """A Driver policy that also considers recommended actions from assistant

    How much the driver follows the assistant's suggestions versus relying on
    the IDM driver model is controlled by the "independence" hyperparameter
    """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 *args,
                 independence: float = 0.9,
                 **kwargs):
        super().__init__(road, position, *args, **kwargs)
        self.independence = independence

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """ Overrides IDMDriverVehicle.get_action() """
        assistant_signal, other_vehicle_obs = self.parse_obs(obs)
        self._update_dynamics(assistant_signal, dt)
        idm_action = self._get_idm_action(other_vehicle_obs)
        action = self._calc_action(assistant_signal[-2:], idm_action)
        return action

    def _calc_action(self,
                     assistant_action: np.ndarray,
                     idm_action: np.ndarray) -> np.ndarray:
        return (
            (1 - self.independence) * assistant_action
            + (self.independence * idm_action)
        )


class ChangingGuidedIDMDriverPolicy(GuidedIDMDriverPolicy):
    """A GuidedIDMDriverPolicy where the driver parameters can be re-sampled

    How much the driver follows the assistant's suggestions versus relying on
    the IDM driver model is controlled by the "independence" hyperparameter
    """

    INDEPENDENCE_DIST = utils.get_truncated_normal(
        0.5, 0.25, 0.0, 1.0
    )

    def __init__(self,
                 road: Road,
                 position: Vector,
                 *args,
                 independence: float = 0.9,
                 **kwargs):
        super().__init__(road, position, *args, **kwargs)
        self.independence = independence

    def get_action(self, obs: np.ndarray, dt: float) -> Action:
        """ Overrides IDMDriverVehicle.get_action() """
        assistant_signal, other_vehicle_obs = self.parse_obs(obs)
        self._update_dynamics(assistant_signal, dt)
        idm_action = self._get_idm_action(other_vehicle_obs)
        action = self._calc_action(assistant_signal[-2:], idm_action)
        return action

    def _calc_action(self,
                     assistant_action: np.ndarray,
                     idm_action: np.ndarray) -> np.ndarray:
        return (
            (1 - self.independence) * assistant_action
            + (self.independence * idm_action)
        )

    @classmethod
    def create_from(cls, vehicle: Vehicle, **kwargs) -> "IDMDriverPolicy":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        driver_config = deepcopy(kwargs)
        driver_config.update(sample_driver_config())

        if kwargs.get("force_independent", False):
            driver_config["independence"] = 1.0
        else:
            driver_config["independence"] = cls.INDEPENDENCE_DIST.rvs()

        return cls(
            road=vehicle.road,
            position=vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            **kwargs
        )


def driver_policy_factory(env, driver_config: dict) -> 'IDMDriverPolicy':
    """Get the driver policy for given driver configuration """
    if driver_config["type"] == "RandomDriverPolicy":
        policy_cls = RandomDriverPolicy
    elif driver_config["type"] == "GuidedIDMDriverPolicy":
        policy_cls = GuidedIDMDriverPolicy
    elif driver_config["type"] == "ChangingGuidedIDMDriverPolicy":
        policy_cls = ChangingGuidedIDMDriverPolicy
    else:
        raise ValueError(f"Unsupported Driver Type: {driver_config['type']}")

    return policy_cls.create_from(env.vehicle, **driver_config)
