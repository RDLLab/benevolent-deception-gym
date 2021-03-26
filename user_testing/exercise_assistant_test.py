"""Script for running Human User Test for Exercise Assistant Env.

The test progresses as follows:
1. Practice Period versus ObedientAthletePolicy for upto PRACTICE_PERIOD min
2. Evaluate user performance versus ObedientAthletePolicy for EVAL_EPISODES
   episodes
3. Practice Period versus IndependentAthletePolicy for upto PRACTICE_PERIOD min
4. Evaluate user performance versus IndependentAthletePolicy for EVAL_EPISODES
   episodes
5. Practice Period versus RandomWeightedAthletePolicy for upto PRACTICE_PERIOD
   min
6. Evaluate user performance versus RandomWeightedAthletePolicy for
   EVAL_EPISODES episodes

All tests are done using discrete action environments
"""
import time
from collections import namedtuple

import gym
import numpy as np

import bdgym    # noqa: F401 pylint: disable=unused-import
from bdgym.envs.exercise_assistant.resources import display_info
from bdgym.envs.exercise_assistant.policy import ManualDiscreteAssistantPolicy

import common


TEST_NAME = "exercise_assistant_test"

PRACTICE_PERIOD = 10    # In Minutes
PRACTICE_PERIOD_SEC = PRACTICE_PERIOD * 60

ALWAYS_SHOW_INFO = True

# Map from test env name to number of evaluation episodes
TEST_ENVS = {
    "ExerciseAssistantOA-v0": 5,
    "ExerciseAssistantIA-v0": 5,
    "ExerciseAssistantHA-v0": 10
}

# Unique user ID
UUID = common.get_uuid()

EvalEpResult = namedtuple(
    "EvalEpResult",
    [
        "source",
        "user_id",
        "env_name",
        "episode_num",
        "ep_return",
        "steps",
        "sets_completed",
        "overexerted",
        "deception_mean",
        "deception_std",
        "ep_time",
        "athlete_independence",
        "athlete_preception_influence",
        "athlete_threshold"
    ]
)


def run_episode(env_name: str, ep_num: int) -> EvalEpResult:
    """Run a single episode """
    print("\n" + common.LINE_BREAK)
    print(f"STARTING EPISODE {ep_num}")
    print(common.LINE_BREAK)
    env = gym.make(env_name)
    assistant_policy = ManualDiscreteAssistantPolicy(False, True)

    obs = env.reset()
    env.render('human')

    total_return = 0.0
    steps = 0
    done = False
    start_time = time.time()
    while not done:
        action = assistant_policy.get_action(obs, env)
        obs, rew, done, _ = env.step(action)
        total_return += rew
        steps += 1

        env.render('human')

    deception_mean = np.mean(env.assistant_deception)
    time_taken = time.time() - start_time
    env.close()

    print(common.SMALL_LINE_BREAK)
    print(f"EPISODE {ep_num} COMPLETE")
    print(common.SMALL_LINE_BREAK)
    print(
        f"return = {total_return:.3f}\n"
        f"steps = {steps}\n"
        f"overexerted = {env.athlete_overexerted()}\n"
        f"mean deception = {deception_mean:.3f}\n"
        f"time = {time_taken:.3f} seconds"
    )
    print(common.SMALL_LINE_BREAK)

    ath_pi = env.athlete_policy
    ep_results = EvalEpResult(
        source=common.SOURCE,
        user_id=UUID,
        env_name=env_name,
        episode_num=ep_num,
        ep_return=total_return,
        steps=steps,
        sets_completed=env.set_count,
        overexerted=env.athlete_overexerted(),
        deception_mean=deception_mean,
        deception_std=np.std(env.assistant_deception),
        ep_time=time_taken,
        athlete_independence=ath_pi.independence,
        athlete_preception_influence=ath_pi.perception_influence,
        athlete_threshold=ath_pi.threshold
    )

    return ep_results


def view_info():
    """View environment information """
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
