"""Tests performance of different Assistant and Athlete fixed policies """
import os.path as osp
from typing import List
import multiprocessing as mp

import bdgym.scripts.exercise_assistant.utils as utils

# (Assistant, Athlete) Fixed Policy pairs
# Note when athlete is 'random' the assistant policy does nothing
POLICY_PAIRS = [
    # ('discrete_donothing', 'weighted'),
    ('discrete_donothing', 'random_weighted'),
    ('discrete_donothing', 'random'),
    # ('discrete_random', 'weighted'),
    # ('discrete_random', 'random_weighted')
]
# Note 0.0 = 'obedient' and 1.0 = 'greedy'
PERCEPT_INFLUENCES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
INDEPENDENCES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
NUM_EPISODES = 1000
SEEDS = list(range(10))
VERBOSE = False
RENDER = ""
MANUAL = False
DISCRETE = True


def create_run_args() -> List[utils.RunArgs]:
    """Create list of arguments for each all runs """
    all_run_args = []
    for policy_pair in POLICY_PAIRS:
        run_args_kwargs = {
            "fixed_athlete_policy": policy_pair[1],
            "fixed_assistant_policy": policy_pair[0],
            "num_episodes": NUM_EPISODES,
            "render": RENDER,
            "no_athlete_render": False,
            "no_assistant_render": False,
            "verbose": VERBOSE,
            "manual": MANUAL,
            "discrete": DISCRETE
        }
        if policy_pair[1] in ('random', 'random_weighted'):
            independences_list = [1.0]
            pi_list = [0.0]
        else:
            independences_list = INDEPENDENCES
            pi_list = PERCEPT_INFLUENCES

        for independence in independences_list:
            for perception_influence in pi_list:
                for seed in SEEDS:
                    run_args = utils.RunArgs(
                        independence=independence,
                        perception_influence=perception_influence,
                        seed=seed,
                        **run_args_kwargs
                    )
                    all_run_args.append(run_args)
    return all_run_args


def run_performance_test():
    """Run the performance test """
    print(f"Running performance test using {utils.NUM_CPUS} cpus")
    all_run_args = create_run_args()

    print(f"Number of runs = {len(all_run_args)}")

    filename = osp.splitext(__file__)[0]

    with mp.Pool(utils.NUM_CPUS) as pool:
        all_results = pool.map(utils.run, all_run_args)

    utils.save_results(all_results, filename, True)


if __name__ == "__main__":
    run_performance_test()
