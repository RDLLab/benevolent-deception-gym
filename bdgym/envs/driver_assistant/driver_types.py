"""Different driver types for the highway autopilot env """
from highway_env.vehicle.controller import ControlledVehicle

import bdgym.envs.utils as utils


MAX_SPEED = 50


def get_driver_config(driver_type: str) -> dict:
    """Get driver config for give driver type """
    driver_type = driver_type.lower()
    if driver_type == "aggressive":
        return AGGRESSIVE_DRIVER
    if driver_type == "standard":
        return STANDARD_DRIVER
    raise ValueError(
        f"Invalid driver type '{driver_type}'. Supported driver"
        f" types are: {AVAILABLE_DRIVER_TYPES}"
    )


AVAILABLE_DRIVER_TYPES = ['aggressive', 'standard']


AGGRESSIVE_DRIVER = {
    "target_speed": MAX_SPEED,
    "acc_max": 15.0,
    "comfort_acc_max": 10.0,
    "comfort_acc_min": -10.0,
    "distance_wanted": 3.0,
    "time_wanted": 0.1,
    "delta": 10.0,
    "politeness": 0,
    "lane_change_min_acc_gain": 0.5,
    "lane_change_max_braking_imposed": 1.0,
    "lane_change_delay": 0.1
}
""" An aggressive driver.

Has higher target speed and acceleration
"""

STANDARD_DRIVER = {
    "target_speed": 30.0,
    "acc_max": 6.0,
    "comfort_acc_max": 3.0,
    "comfort_acc_min": -5.0,
    "distance_wanted": 5.0 + ControlledVehicle.LENGTH,
    "time_wanted": 1.5,
    "delta": 4.0,
    "politeness": 0.0,
    "lane_change_min_acc_gain": 0.2,
    "lane_change_max_braking_imposed": 2.0,
    "lane_change_delay": 1.0
}
""" The default IDM driver.

Uses default values from the Highway Env IDMVehicle
"""

DRIVER_PARAM_LIMITS = {
    "target_speed": [20, MAX_SPEED],
    "acc_max": [3.0, 15.0],
    "comfort_acc_max": [1.0, 10.0],
    "comfort_acc_min": [-10, -5.0],
    "distance_wanted": [3.0, 2*ControlledVehicle.LENGTH],
    "time_wanted": [0.1, 2.0],
    "delta": [2.0, 10.0],
    "politeness": [0.0, 1.0],
    "lane_change_min_acc_gain": [0.1, 0.5],
    "lane_change_max_braking_imposed": [1.0, 3.0],
    "lane_change_delay": [0.1, 2.0]
}


def sample_driver_config() -> dict:
    """Sample a driver configuration.

    Will sample each parameter from a truncated normal distribution with
    mean=0.5, var=0.25 bounded to (0.0, 1.0) then mapped to the parameter
    range as specified in DRIVER_PARAM_LIMITS.

    Returns
    -------
    dict
        the sampled driver config
    """
    driver_config = {}
    normal_dist = utils.get_truncated_normal(0.5, 0.25, 0.0, 1.0)
    for param, value_range in DRIVER_PARAM_LIMITS.items():
        norm_sample = normal_dist.rvs()
        param_value = utils.lmap(norm_sample, [0.0, 1.0], value_range)
        driver_config[param] = param_value
    return driver_config
