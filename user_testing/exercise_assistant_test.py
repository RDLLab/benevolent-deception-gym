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
import os
import os.path as osp
import time
import datetime
from collections import namedtuple
from typing import List, Union, Tuple

import gym
import numpy as np

import bdgym    # noqa: F401 pylint: disable=unused-import
from bdgym.envs.exercise_assistant.resources import display_keybindings
from bdgym.envs.exercise_assistant.policy import ManualDiscreteAssistantPolicy


PRACTICE_PERIOD = 10    # In Minutes
PRACTICE_PERIOD_SEC = PRACTICE_PERIOD * 60

# Map from test env name to number of evaluation episodes
TEST_ENVS = {
    "ExerciseAssistantOA-v0": 3,
    "ExerciseAssistantIA-v0": 3,
    "ExerciseAssistantHA-v0": 5
}

LINE_BREAK = "=" * 60
SMALL_LINE_BREAK = "-" * 30

BASE_RESULTS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "results")

if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)


EvalEpResult = namedtuple(
    "EvalEpResult",
    [
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


def write_results_to_file(results: Union[List[EvalEpResult], EvalEpResult],
                          filepath: str):
    """Write result to file

    Will add header if file at filepath doesn't exist
    """
    if not isinstance(results, list):
        results = [results]

    if not osp.isfile(filepath):
        with open(filepath, "w") as fout:
            headers = EvalEpResult._fields
            fout.write("\t".join(headers) + "\n")

    with open(filepath, "a") as fout:
        for result in results:
            row = []
            for v in result._asdict().values():
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            fout.write("\t".join(row) + "\n")


def create_subdir(subdir_name: str) -> str:
    """Create a new subdirectory with timestamp appended to name """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    subdir_name = f"{subdir_name}_{timestamp}.tsv"
    full_dirname = osp.join(BASE_RESULTS_DIR, subdir_name)

    if not osp.exists(full_dirname):
        os.makedirs(full_dirname)
    return full_dirname


def run_episode(env_name: str, ep_num: int) -> EvalEpResult:
    """Run a single episode """
    print("\n" + LINE_BREAK)
    print(f"STARTING EPISODE {ep_num}")
    print(LINE_BREAK)
    env = gym.make(env_name)
    assistant_policy = ManualDiscreteAssistantPolicy(False, True)

    obs = env.reset()
    env.render('human')

    total_return = 0.0
    steps = 0
    done = False
    start_time = time.time()
    while not done:
        print(SMALL_LINE_BREAK)
        print(f"ASSISTANT Step = {steps}")
        print(SMALL_LINE_BREAK)
        action = assistant_policy.get_action(obs, env)
        obs, rew, done, _ = env.step(action)
        total_return += rew

        print(f"\nReward: {rew}")
        print(f"Done: {done}")
        steps += 1

        env.render('human')

    deception_mean = np.mean(env.assistant_deception)
    time_taken = time.time() - start_time
    env.close()

    print(SMALL_LINE_BREAK)
    print(f"EPISODE {ep_num} COMPLETE")
    print(SMALL_LINE_BREAK)
    print(
        f"return = {total_return:.3f}\n"
        f"steps = {steps}\n"
        f"overexerted = {env.athlete_overexerted()}\n"
        f"mean deception = {deception_mean:.3f}\n"
        f"time = {time_taken:.3f} seconds"
    )
    print(SMALL_LINE_BREAK)

    ath_pi = env.athlete_policy
    ep_results = EvalEpResult(
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


def run_env_eval(env_name: str, results_file: str) -> List[EvalEpResult]:
    """Run evaluation of an environment """
    eval_episodes = TEST_ENVS[env_name]
    print("\n" + LINE_BREAK)
    print(f"Starting Evaluation for environment: {env_name}")
    print(LINE_BREAK)
    print(f"Number of evaluation episodes = {eval_episodes}")

    ep_results = []
    for e in range(eval_episodes):
        input(f"Press any ENTER to begin evaluation episode {e+1}.")
        print()
        result = run_episode(env_name, e)
        ep_results.append(result)
        write_results_to_file(result, results_file)

    print(LINE_BREAK)
    print(f"Evaluation complete for environment: {env_name}")
    print(LINE_BREAK)

    return ep_results


def run_env_practice(env_name: str, results_file: str) -> List[EvalEpResult]:
    """Run practice for an environment """
    print("\n" + LINE_BREAK)
    print(f"Starting Practice for environment: {env_name}")
    print(LINE_BREAK)
    print(
        f"You can practice for up to {PRACTICE_PERIOD} minutes on this "
        "environment."
    )
    print(
        "If you reach the time limit you will be allowed to complete your "
        "current episode."
    )
    print(
        "After each episode you will be asked if you wish to end the practice "
        "early."
    )

    start_time = time.time()
    ep_num = 0
    ep_results = []
    while time.time() - start_time < PRACTICE_PERIOD_SEC:
        input(f"Press any ENTER to begin practice episode {ep_num+1}.")
        print()
        result = run_episode(env_name, ep_num)
        ep_results.append(result)
        write_results_to_file(result, results_file)
        ep_num += 1

        print(LINE_BREAK)
        time_left = PRACTICE_PERIOD_SEC - (time.time() - start_time)
        if time_left < 0:
            print("Time limit reached. Ending practice")
            break

        min_left = int(time_left // 60)
        sec_left = int(time_left % 60)
        print(f"You have {min_left}:{sec_left} min left for practice")
        answer = input("Would you like to continue practicing [y]/N: ")
        print()
        if answer.lower() == "n":
            print("Ending practice early")
            break

        print("Continuing practice")

    return ep_results


def run_env_user_test(env_name: str,
                      results_dir: str
                      ) -> Tuple[List[EvalEpResult], List[EvalEpResult]]:
    """Run user test on environment, including practice and evaluation"""
    print("\n" + LINE_BREAK)
    print(f"Starting User Testing for environment: {env_name}")
    print(LINE_BREAK)
    print(
        f"This will involve a practice period of {PRACTICE_PERIOD} min "
        f"followed by an evaluation over {TEST_ENVS[env_name]} episodes."
    )
    input("Press any ENTER to begin practice period.")

    practice_results_file = osp.join(results_dir, f"{env_name}_practice")
    practice_results = run_env_practice(env_name, practice_results_file)

    print(LINE_BREAK)
    input("Practice period complete. Press ENTER to begin evaluation.")

    eval_results_file = osp.join(results_dir, f"{env_name2}_eval")
    eval_results = run_env_eval(env_name, eval_results_file)

    print(LINE_BREAK)
    print(f"User Testing for {env_name} environment complete.")
    input("Press any ENTER to move onto next testing stage.")
    print(LINE_BREAK)

    return practice_results, eval_results


def view_keybindings():
    """View keybindings """
    answer = input(
        "Would you like to view the keybindings for the environment [y]/N? "
    )
    if answer.lower() == "n":
        return
    display_keybindings()


def run_user_test():
    """Run the full user test """
    print(LINE_BREAK)
    print("Starting User Test")
    print(LINE_BREAK)
    print(f"Testing will be done for {len(TEST_ENVS)} environments.")
    print(
        "For each, the user will have a practice period followed by the "
        "evaluation."
    )
    print(
        "There will be opportunities to have a break before each practice and "
        "evaluation period."
    )
    view_keybindings()
    input("Press any ENTER to begin the testing.")

    results_dir = create_subdir("exercise_assistant_test")
    for env_name in TEST_ENVS:
        run_env_user_test(env_name, results_dir)

    print(LINE_BREAK)
    print("User Testing for for all environments complete!.")
    print("Thank you for being a tester :)")
    print(
        "Once all results are collected we will let you know how you went "
        "compared to other testers. Purely for bragging purposes."
    )
    print(LINE_BREAK)


if __name__ == "__main__":
    run_user_test()
