"""Stable Baseline3 PPO implementation for BDGym Highway autopilot env """
import os.path as osp
from pprint import pprint

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import bdgym.scripts.drl_utils as drl_utils
from bdgym.envs.driver_assistant.fixed_driver_env import \
    FixedDriverDriverAssistantEnv


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)
FILENAME = osp.splitext(__file__)[0]


# TODO
# Handle continuous vs discrete assistant actions
# update the configuration to be correct


def get_config_env(args, seed=0):
    """Get the configured env """
    env = FixedDriverDriverAssistantEnv()
    env.seed(seed)

    env_obs = env.config["observation"]
    env_obs["stack_size"] = args.stack_size

    env.configure({
        "observation": env_obs,
        "driver": {
            "type": args.driver_type,
            "independence": args.independence
        },
        "offroad_terminal": True
    })

    env.reset()
    return env


def make_env(args, rank, seed=0):
    """Utility function for multiprocessed env. """
    def _init():
        env = get_config_env(args, seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_env(args, eval_env=False):
    """Get the agent env """
    if args.num_cpus == 1:
        print("Using single environment")
        env = get_config_env(args)
    elif eval_env:
        env = SubprocVecEnv([make_env(args, i) for i in range(1)])
    else:
        print(f"Running {args.num_cpus} envs in parallel")
        env = SubprocVecEnv([make_env(args, i) for i in range(args.num_cpus)])
    return env


def main(args):
    """train the agent """
    assert args.save_frequency == -1 \
        or 0 <= args.save_frequency <= args.total_timesteps \
        or args.save_frequency >= args.batch_steps * args.num_cpus

    print("Running baseline3 PPO using args:")
    pprint(args)

    env = get_env(args)
    if args.load_model == "":
        ppo_model = drl_utils.init_ppo_model(
            env,
            batch_steps=args.batch_steps,
            verbosity=args.verbose,
            result_dir=RESULTS_DIR,
            policy='MlpPolicy',
            **{'gamma': 0.999, "ent_coef": 0.1}
        )
    else:
        ppo_model = drl_utils.load_ppo_model(args.load_model, env)

    drl_utils.run_model(ppo_model, get_env(args, True), args.verbosity)

    reset_num_timesteps = args.load_model == ""
    training_complete = args.total_timesteps <= 0
    while not training_complete:
        eval_env = get_env(args, True)
        drl_utils.train_model(
            ppo_model,
            total_timesteps=args.total_timesteps,
            eval_env=eval_env,
            save_frequency=args.save_frequency,
            eval_freq=args.batch_steps,
            reset_num_timesteps=reset_num_timesteps,
            log_name=f"{FILENAME}_DriverAssistantEnv"
        )
        reset_num_timesteps = False
        drl_utils.run_model(
            ppo_model, get_env(args, True), args.verbosity, wait_for_user=True
        )
        training_complete = not drl_utils.continue_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--independence",  type=float, default=0.0,
                        help="driver independence (default=0.0)")
    parser.add_argument("-d", "--driver_type", type=str,
                        default="GuidedIDMDriverVehicle",
                        help="driver type (default='GuidedIDMDriverVehicle'")
    parser.add_argument("-f", "--simulation_frequency", type=int, default=15,
                        help="env simulation frequency (default=15)")
    parser.add_argument("-ss", "--stack_size", type=int, default=1,
                        help="number of observations to stack (default=1)")
    parser = drl_utils.get_ppo_argparse(parser)
    main(parser.parse_args())
