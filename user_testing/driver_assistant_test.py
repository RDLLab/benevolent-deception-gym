"""Script for running Human User Test for Driver Assistant Env.

The user is tested on 4 different versions of the Driver Assistant Env:
1. Obedient Driver
2. Independent Driver
3. Aggressive Independent Driver
4. Human Driver (i.e. parameters are different each episode)

For each version the test progresses as follows:
1. Practice Period versus for up to PRACTICE_PERIOD min
2. Evaluate user performance for a fixed number of episodes (See TEST_ENVS for
   the # eval episodes for each of the environment versions)

All tests are done using discrete action environments
"""
import time
from collections import namedtuple

import gym
import numpy as np

import bdgym    # noqa: F401 pylint: disable=unused-import
from bdgym.envs.driver_assistant.resources import display_info
from bdgym.envs.driver_assistant.driver_types import DRIVER_PARAM_LIMITS
from bdgym.envs.driver_assistant.manual_control import AssistantEventHandler

import common


TEST_NAME = "driver_assistant_test"

PRACTICE_PERIOD = 20    # In Minutes
PRACTICE_PERIOD_SEC = PRACTICE_PERIOD * 60

ALWAYS_SHOW_INFO = True

# Map from test env name to number of evaluation episodes
TEST_ENVS = {
    "DriverAssistantOD-v0": 10,
    "DriverAssistantID-v0": 10,
    "DriverAssistantAggressiveID-v0": 10,
    "DriverAssistantHD-v0": 10
}


# Unique user ID
UUID = common.get_uuid()

results_params = [
    "source",
    "user_id",
    "env_name",
    "episode_num",
    "ep_return",
    "steps",
    "crashed",
    "deception_x_mean",
    "deception_x_std",
    "deception_y_mean",
    "deception_y_std",
    "deception_vx_mean",
    "deception_vx_std",
    "deception_vy_mean",
    "deception_vy_std",
    "ep_time",
    "independence",
]
results_params.extend(list(DRIVER_PARAM_LIMITS))
DriverEvalEpResult = namedtuple("DriverEvalEpResult", results_params)


def run_episode(env_name: str, ep_num: int) -> DriverEvalEpResult:
    """Run a single episode """
    print("\n" + common.LINE_BREAK)
    print(f"STARTING EPISODE {ep_num}")
    print(common.LINE_BREAK)
    env = gym.make(env_name)
    env.config["manual_control"] = True

    env.reset()
    env.render('human')

    total_return = 0.0
    steps = 0
    done = False
    start_time = time.time()
    while not done:
        action = AssistantEventHandler.get_discrete_action(env)
        _, rew, done, _ = env.step(action)
        total_return += rew
        steps += 1
        env.render('human')

    deception_mean = np.mean(env.assistant_deception, axis=0)
    deception_std = np.std(env.assistant_deception, axis=0)
    crashed = steps < env.config["duration"]
    time_taken = time.time() - start_time
    env.close()

    print(common.SMALL_LINE_BREAK)
    print(f"EPISODE {ep_num} COMPLETE")
    print(common.SMALL_LINE_BREAK)
    print(
        f"return = {total_return:.3f}\n"
        f"steps = {steps}\n"
        f"crashed = {crashed}\n"
        f"mean x deception = {deception_mean[0]:.3f}\n"
        f"mean y deception = {deception_mean[1]:.3f}\n"
        f"mean vx deception = {deception_mean[2]:.3f}\n"
        f"mean vy deception = {deception_mean[3]:.3f}\n"
        f"time = {time_taken:.3f} seconds"
    )
    print(common.SMALL_LINE_BREAK)

    driver_policy_params = env.driver_policy.configuration
    ep_results = DriverEvalEpResult(
        source=common.SOURCE,
        user_id=UUID,
        env_name=env_name,
        episode_num=ep_num,
        ep_return=total_return,
        steps=steps,
        crashed=crashed,
        deception_x_mean=float(deception_mean[0]),
        deception_x_std=float(deception_std[0]),
        deception_y_mean=float(deception_mean[1]),
        deception_y_std=float(deception_std[1]),
        deception_vx_mean=float(deception_mean[2]),
        deception_vx_std=float(deception_std[2]),
        deception_vy_mean=float(deception_mean[3]),
        deception_vy_std=float(deception_std[3]),
        ep_time=time_taken,
        **driver_policy_params
    )

    return ep_results


def view_info():
    """View keybindings """
    if ALWAYS_SHOW_INFO:
        display_info()
        return

    answer = input(
        "Would you like to view the info for the environment [y]/N? "
    )
    if answer.lower() != "n":
        display_info()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Name of tester")
    args = parser.parse_args()
    UUID = args.name
    common.run_user_test(
        f"{TEST_NAME}_{args.name}",
        TEST_ENVS,
        PRACTICE_PERIOD,
        run_episode,
        view_info
    )
