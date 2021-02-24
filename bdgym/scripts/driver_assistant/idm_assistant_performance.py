"""Test performance of all combos of IDM driver and autopilot """
import os
import os.path as osp
import datetime
from typing import List
import multiprocessing as mp

import bdgym.scripts.driver_assistant.utils as utils
from bdgym.envs.driver_assistant.driver_types import AVAILABLE_DRIVER_TYPES


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)
FILENAME = osp.splitext(__file__)[0]

NUM_CPUS = max(1, len(os.sched_getaffinity(0)) - 4)

ASSISTANTS = AVAILABLE_DRIVER_TYPES + ['random']
DRIVERS = AVAILABLE_DRIVER_TYPES
INDEPENDENCES = [0.0, 0.5, 1.0]
NUM_EPISODES = 100
SEED = 0
NORMALIZE_OBS = False
VERBOSE = False
RENDER = False
MANUAL = False
TIME_DELAY = 0.0


def create_run_args() -> List[utils.RunArgs]:
    """Create list of arguments for each all runs """
    all_run_args = []
    for assistant in ASSISTANTS:
        for driver in DRIVERS:
            for independence in INDEPENDENCES:
                run_args = utils.RunArgs(
                    independence=independence,
                    driver_type=driver,
                    assistant_type=assistant,
                    num_episodes=NUM_EPISODES,
                    seed=SEED,
                    render=RENDER,
                    normalize_obs=NORMALIZE_OBS,
                    verbose=VERBOSE,
                    manual=MANUAL,
                    time_delay=TIME_DELAY
                )
                all_run_args.append(run_args)
    return all_run_args


def save_results(all_results: List[utils.Result]):
    """Save all results to file """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = osp.join(RESULTS_DIR, f"{FILENAME}_{timestamp}.csv")

    print(f"\nSaving results to: {filename}")

    with open(filename, "w") as fout:
        headers = all_results[0]._fields
        fout.write("\t".join(headers) + "\n")
        for result in all_results:
            row = [str(v) for v in result._asdict().values()]
            fout.write("\t".join(row) + "\n")


def run_performance_test():
    """Run the performance test """
    print(f"Running performance test using {NUM_CPUS} cpus")
    all_run_args = create_run_args()

    print(f"Number of runs = {len(all_run_args)}")

    with mp.Pool(NUM_CPUS) as pool:
        all_results = pool.map(utils.run, all_run_args)

    save_results(all_results)


if __name__ == "__main__":
    run_performance_test()
