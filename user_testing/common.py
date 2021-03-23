"""Common functions for user testing """
import os
import os.path as osp
import time
import uuid
import datetime
from typing import List, Union, Tuple, NamedTuple, Dict

BASE_RESULTS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "results")

if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

LINE_BREAK = "=" * 60
SMALL_LINE_BREAK = "-" * 30

SOURCE = "human"


def get_uuid() -> str:
    """Get a unique User ID """
    return str(uuid.uuid1())


def write_results_to_file(results: Union[List[NamedTuple], NamedTuple],
                          filepath: str):
    """Write result to file

    Will add header if file at filepath doesn't exist
    """
    if not isinstance(results, list):
        results = [results]

    if not osp.isfile(filepath):
        with open(filepath, "w") as fout:
            headers = results[0]._fields
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


def run_env_eval(env_name: str,
                 eval_episodes: int,
                 results_file: str,
                 run_episode_fn) -> List[NamedTuple]:
    """Run evaluation of an environment """
    print("\n" + LINE_BREAK)
    print(f"Starting Evaluation for environment: {env_name}")
    print(LINE_BREAK)
    print(f"Number of evaluation episodes = {eval_episodes}")

    ep_results = []
    for e in range(eval_episodes):
        input(f"Press any ENTER to begin evaluation episode {e+1}.")
        print()
        result = run_episode_fn(env_name, e)
        ep_results.append(result)
        write_results_to_file(result, results_file)

    print(LINE_BREAK)
    print(f"Evaluation complete for environment: {env_name}")
    print(LINE_BREAK)

    return ep_results


def run_env_practice(env_name: str,
                     practice_period: int,
                     results_file: str,
                     run_episode_fn) -> List[NamedTuple]:
    """Run practice for an environment """
    print("\n" + LINE_BREAK)
    print(f"Starting Practice for environment: {env_name}")
    print(LINE_BREAK)
    print(
        f"You can practice for up to {practice_period} minutes on this "
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
    practice_period_sec = practice_period * 60
    ep_num = 0
    ep_results = []
    while time.time() - start_time < practice_period_sec:
        input(f"Press any ENTER to begin practice episode {ep_num+1}.")
        print()
        result = run_episode_fn(env_name, ep_num)
        ep_results.append(result)
        write_results_to_file(result, results_file)
        ep_num += 1

        print(LINE_BREAK)
        time_left = practice_period_sec - (time.time() - start_time)
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
                      practice_period: int,
                      eval_episodes: int,
                      results_dir: str,
                      run_episode_fn,
                      ) -> Tuple[List[NamedTuple], List[NamedTuple]]:
    """Run user test on environment, including practice and evaluation"""
    print("\n" + LINE_BREAK)
    print(f"Starting User Testing for environment: {env_name}")
    print(LINE_BREAK)
    print(
        f"This will involve a practice period of {practice_period} min "
        f"followed by an evaluation over {eval_episodes} episodes."
    )
    input("Press any ENTER to begin practice period.")

    practice_results_file = osp.join(results_dir, f"{env_name}_practice")
    practice_results = run_env_practice(
        env_name, practice_period, practice_results_file, run_episode_fn
    )

    print(LINE_BREAK)
    input("Practice period complete. Press ENTER to begin evaluation.")

    eval_results_file = osp.join(results_dir, f"{env_name}_eval")
    eval_results = run_env_eval(
        env_name, eval_episodes, eval_results_file, run_episode_fn
    )

    print(LINE_BREAK)
    print(f"User Testing for {env_name} environment complete.")
    input("Press any ENTER to move onto next testing stage.")
    print(LINE_BREAK)

    return practice_results, eval_results


def run_user_test(test_name: str,
                  test_envs: Dict[str, int],
                  practice_period: int,
                  run_episode_fn,
                  view_info_fn):
    """Run the full user test """
    start_time = time.time()
    print(LINE_BREAK)
    print("Starting User Test")
    print(LINE_BREAK)
    print(f"Testing will be done for {len(test_envs)} environments.")
    print(
        "For each, the user will have a practice period followed by the "
        "evaluation."
    )
    print(
        "There will be opportunities to have a break before each practice and "
        "evaluation period."
    )
    view_info_fn()
    input("Press any ENTER to begin the testing.")

    results_dir = create_subdir(test_name)
    for env_name, eval_episodes in test_envs.items():
        run_env_user_test(
            env_name,
            practice_period,
            eval_episodes,
            results_dir,
            run_episode_fn
        )

    time_taken = time.time() - start_time
    min_taken = int(time_taken // 60)
    sec_taken = int(time_taken % 60)

    print(LINE_BREAK)
    print("User Testing for for all environments complete!.")
    print(f"Total test time = {min_taken}:{sec_taken} min")
    print("Thank you for being a tester :)")
    print(
        "Once all results are collected we will let you know how you went "
        "compared to other testers. Purely for bragging purposes."
    )
    print(LINE_BREAK)
