"""StableBaselines3 PPO implementation for the Exercise Assistant Env

For Fixed Athlete Policy.
"""
import os.path as osp
from pprint import pprint

import bdgym.scripts.drl_utils as drl_utils
import bdgym.envs.exercise_assistant as ea_env
import bdgym.envs.exercise_assistant.policy as policy
import bdgym.scripts.exercise_assistant.utils as utils


def get_config_env(args,
                   seed: int = 0) -> ea_env.FixedAthleteExerciseAssistantEnv:
    """Get the configured Fixed Athlete Exercise Assistant Env """
    athlete_policy = policy.WeightedAthletePolicy(
        independence=args.independence
    )
    if args.continuous:
        env = ea_env.FixedAthleteExerciseAssistantEnv(athlete_policy)
    else:
        env = ea_env.DiscreteFixedAthleteExerciseAssistantEnv(athlete_policy)

    env.seed(seed)
    return env


def get_env_name(args) -> str:
    """Get the name of the environment being run """
    env = get_config_env(args)
    return env.__class__.__name__


def get_env_creation_fn(args):
    """Get the get_configured_env_fn """
    def get_config_env_fn(seed):
        return get_config_env(args, seed)
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
            verbosity=args.verbosity,
            result_dir=utils.RESULTS_DIR,
            policy='MlpPolicy',
            **{'gamma': 0.999, "ent_coef": 0.1}
        )
    else:
        ppo_model = drl_utils.load_ppo_model(args.load_model, env)

    drl_utils.run_model(
        ppo_model,
        drl_utils.get_env(get_config_env_fn, True),
        args.verbosity
    )

    this_filename = osp.splitext(__file__)[0]
    log_name = (
        f"{this_filename}_{get_env_name(args)}_i{args.independence:.3f}"
    )

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
            log_name=log_name
        )
        reset_num_timesteps = False
        drl_utils.run_model(
            ppo_model,
            drl_utils.get_env(get_config_env_fn, True),
            args.verbosity,
            wait_for_user=True
        )
        training_complete = not drl_utils.continue_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--independence",  type=float, default=0.0,
                        help="athlete independence (default=0.0)")
    # parser.add_argument("-ss", "--stack_size", type=int, default=1,
    #                     help="number of observations to stack (default=1)")
    parser.add_argument("-c", "--continuous", action="store_true",
                        help="use continuous actions")
    parser = drl_utils.get_ppo_argparse(parser)
    main(parser.parse_args())
