"""Stable Baseline3 PPO implementation for BDGym Highway autopilot env """
import os.path as osp
from pprint import pprint

from stable_baselines3 import PPO

import bdgym.scripts.drl_utils as drl_utils
import bdgym.scripts.driver_assistant.utils as utils
from bdgym.envs.driver_assistant.fixed_driver_env import \
    FixedDriverDriverAssistantEnv


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)
FILENAME = osp.splitext(__file__)[0]


# TODO
# Handle continuous vs discrete assistant actions


def get_env_creation_fn(args):
    """Get the get_configured_env_fn """
    def get_config_env_fn(seed):
        kwargs = vars(args)
        kwargs['seed'] = seed
        return utils.get_configured_env(**kwargs)
    return get_config_env_fn


def main(args):
    """train the agent """
    assert args.save_frequency == -1 \
        or 0 <= args.save_frequency <= args.total_timesteps \
        or args.save_frequency >= args.batch_steps * args.num_cpus

    print("Running baseline3 PPO using args:")
    pprint(args)

    get_config_env_fn = get_env_creation_fn(args)
    env = drl_utils.get_env(get_config_env_fn, False, args.num_cpus, args.seed)
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
        ppo_model = drl_utils.load_model(PPO, args.load_model, env)

    drl_utils.run_model(
        ppo_model, drl_utils.get_env(get_config_env_fn, True), args.verbosity
    )

    this_filename = osp.splitext(__file__)[0]
    env_name = utils.get_env_name(args)
    log_name = (
        f"{this_filename}_{env_name}_i{args.independence:.3f}"
    )
    result_dir = osp.join(utils.RESULTS_DIR, log_name)

    reset_num_timesteps = args.load_model == ""
    training_complete = args.total_timesteps <= 0
    while not training_complete:
        eval_env = drl_utils.get_env(get_config_env_fn, True)
        drl_utils.train_model(
            ppo_model,
            total_timesteps=args.total_timesteps,
            eval_env=eval_env,
            save_frequency=args.save_frequency,
            eval_freq=args.batch_steps,
            reset_num_timesteps=reset_num_timesteps,
            log_name=f"{FILENAME}_DriverAssistantEnv",
            save_best=args.save_best,
            result_dir=result_dir
        )
        reset_num_timesteps = False
        drl_utils.run_model(
            ppo_model,
            drl_utils.get_env(get_config_env_fn, True),
            args.verbosity,
            wait_for_user=True
        )
        training_complete = not drl_utils.continue_training()

    if args.save_best:
        print("Running Best model")
        env = drl_utils.get_env(get_config_env_fn, True)
        best_model = drl_utils.load_best_model(PPO, result_dir, env)
        drl_utils.run_model(
            best_model,
            drl_utils.get_env(get_config_env_fn, True),
            args.verbosity,
            wait_for_user=True
        )


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
    parser.add_argument("-dc", "--discrete", action="store_true",
                        help="use discrete assistant actions")
    parser.add_argument("-fi", "--force_independent", action="store_true",
                        help=("Ensure driver is independent "
                              "(mainly used for 'changing' driver type)"))
    parser = drl_utils.get_ppo_argparse(parser)
    main(parser.parse_args())
