"""Test performance of all combos of IDM driver and autopilot """
import os.path as osp
from typing import List
import multiprocessing as mp

import bdgym.scripts.script_utils as script_utils
import bdgym.scripts.driver_assistant.utils as utils
from bdgym.envs.driver_assistant.driver_types import AVAILABLE_DRIVER_TYPES


ASSISTANTS = ['random']
DRIVERS = AVAILABLE_DRIVER_TYPES + ['random', 'changing']
INDEPENDENCES = [1.0]
NUM_EPISODES = 100
SEED = 0
NORMALIZE_OBS = False
VERBOSITY = 1
RENDER = False
MANUAL = False
DISCRETE = True
FORCE_INDEPENDENT = True
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
                    verbosity=VERBOSITY,
                    manual=MANUAL,
                    discrete=DISCRETE,
                    force_independent=FORCE_INDEPENDENT,
                    time_delay=TIME_DELAY
                )
                all_run_args.append(run_args)
    return all_run_args


def run_performance_test():
    """Run the performance test """
    print(f"Running performance test using {utils.NUM_CPUS} cpus")
    all_run_args = create_run_args()

    print(f"Number of runs = {len(all_run_args)}")

    with mp.Pool(utils.NUM_CPUS) as pool:
        all_results = pool.map(utils.run, all_run_args)

    filename = osp.splitext(__file__)[0]
    script_utils.save_results(
        all_results, utils.RESULTS_DIR, filename, True
    )


if __name__ == "__main__":
    run_performance_test()
