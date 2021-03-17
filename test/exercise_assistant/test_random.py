"""Runs random agents on environment """
from importlib import reload

import gym
import pytest

import bdgym
import bdgym.envs.exercise_assistant as eagym


TEST_SEEDS = [0, 1, 5, 10, 666]


def test_gym_reload():
    """Tests there is no issue when reloading gym """
    reload(gym)
    reload(bdgym)


@pytest.mark.parametrize('env_name', eagym.EXERCISE_ASSISTANT_GYM_ENVS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_multiagent_random(env_name: str, seed: int):
    """Tests Exercise-Assistant multi-agent envs with random agents """
    env = gym.make(env_name)
    env.seed(seed)

    athlete_action_space = env.athlete_action_space
    athlete_action_space.seed(seed)
    assistant_action_space = env.assistant_action_space
    assistant_action_space.seed(seed)

    done = False
    while not done:
        assistant_action = assistant_action_space.sample()
        _, _, done, _ = env.step(assistant_action)

        athlete_action = athlete_action_space.sample()
        _, _, done, _ = env.step(athlete_action)


@pytest.mark.parametrize(
    'env_name', eagym.FIXED_ATHLETE_EXERCISE_ASSISTANT_GYM_ENVS
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_fixed_athlete_random(env_name: str, seed: int):
    """Tests Exercise-Assistant fixed athlete envs with random assistant """
    env = gym.make(env_name)
    env.seed(seed)

    assistant_action_space = env.action_space
    assistant_action_space.seed(seed)

    done = False
    while not done:
        assistant_action = assistant_action_space.sample()
        _, _, done, _ = env.step(assistant_action)
