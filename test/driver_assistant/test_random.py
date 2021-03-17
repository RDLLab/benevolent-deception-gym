"""Runs random agents on environment """
from importlib import reload

import gym
import pytest

import bdgym
import bdgym.envs.driver_assistant as dagym


TEST_SEEDS = [0, 1, 666]


def test_gym_reload():
    """Tests there is no issue when reloading gym """
    reload(gym)
    reload(bdgym)


@pytest.mark.parametrize('env_name', dagym.DRIVER_ASSISTANT_GYM_ENVS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_multiagent_random(env_name: str, seed: int):
    """Tests Driver-Assistant multi-agent envs with random agents """
    env = gym.make(env_name)
    env.seed(seed)

    driver_action_space = env.driver_action_space
    driver_action_space.seed(seed)
    assistant_action_space = env.assistant_action_space
    assistant_action_space.seed(seed)

    done = False
    while not done:
        assistant_action = assistant_action_space.sample()
        _, _, done, _ = env.step(assistant_action)

        driver_action = driver_action_space.sample()
        _, _, done, _ = env.step(driver_action)


@pytest.mark.parametrize(
    'env_name', dagym.FIXED_DRIVER_DRIVER_ASSISTANT_GYM_ENVS
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_fixed_driver_random(env_name: str, seed: int):
    """Tests Driver-Assistant fixed driver envs with random assistant """
    env = gym.make(env_name)
    env.seed(seed)

    assistant_action_space = env.action_space
    assistant_action_space.seed(seed)

    done = False
    while not done:
        assistant_action = assistant_action_space.sample()
        _, _, done, _ = env.step(assistant_action)
