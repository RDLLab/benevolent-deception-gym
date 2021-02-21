"""Stable Baseline3 PPO implementation for BDGym Highway autopilot env """
import os.path as osp
from pprint import pprint

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed, logger

from bdgym.envs.driver_assistant.autopilot_env import \
    HighwayAutopilotEnv


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)
FILENAME = osp.splitext(__file__)[0]


def get_config_env(args, seed=0):
    """Get the configured env """
    env = HighwayAutopilotEnv()
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


def run_model(model, args, wait_for_user=False):
    """Run model on env """
    print("Running episode of the model")
    if wait_for_user:
        input("Press Enter to run episode")

    env = get_config_env(args, seed=None)
    obs = env.reset()
    if args.verbose > 1:
        print(f"\nObservation=\n{np.array_str(obs, precision=3)}")
    done = False
    total_return = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if args.verbose > 1:
            print(
                f"\nAction={np.array_str(action, precision=3)}"
                f"\nObservation=\n{np.array_str(obs, precision=3)}"
                f"\nReward={reward:.3f}"
                f"\nDone={done}"
            )
        total_return += reward
        env.render()
        if args.verbose > 2:
            input()
    print(f"Return = {total_return}")
    env.close()


def get_train_epoch_lengths(args):
    """Get the lengths of learning epochs. Here One epoch is the number
    of timesteps between when the model is saved.
    """
    if args.save_frequency <= 0:
        epoch_lengths = [args.total_timesteps]
    else:
        epoch_lengths = []
        remaining_timesteps = args.total_timesteps
        while remaining_timesteps > 0:
            epoch_lengths.append(min(remaining_timesteps, args.save_frequency))
            remaining_timesteps -= args.save_frequency
    return epoch_lengths


def save_model(ppo_model):
    """Save model """
    print("Saving model")
    model_save_path = logger.get_dir()
    ppo_model.save(model_save_path)
    print(f"Model saved to = {model_save_path}")


def train_model(ppo_model, args, reset_num_timesteps):
    """Run a single training cycle of the PPO model """
    print(f"Training model for {args.total_timesteps} steps")
    eval_env = get_env(args, True)
    epoch_lengths = get_train_epoch_lengths(args)
    print(f"With epochs of sizes: {epoch_lengths}")

    for i, t in enumerate(epoch_lengths):
        print(f"Starting epoch {i+1} of {len(epoch_lengths)}")
        ppo_model.learn(
            total_timesteps=t,
            eval_env=eval_env,
            eval_freq=args.batch_steps,
            n_eval_episodes=10,
            tb_log_name=f"{FILENAME}_HighwayAutopilotEnv",
            reset_num_timesteps=reset_num_timesteps
        )
        reset_num_timesteps = False
        if args.save_frequency != 0:
            save_model(ppo_model)


def continue_training():
    """Check whether to continue training or not """
    answer = input("Continue Training [y/n]?: ")
    if answer.lower() != "y":
        print("Ending training")
        return False
    print("Running another round of learning")
    return True


def init_model(args):
    """Initialize the PPO model """
    env = get_env(args)

    if args.load_model == "":
        ppo_model = PPO(
            'MlpPolicy',
            env,
            n_steps=args.batch_steps,
            verbose=args.verbose,
            tensorboard_log=RESULTS_DIR,
            ent_coef=0.01,
            gamma=0.999,

        )
    else:
        ppo_model = PPO.load(
            args.load_model,
            env=env
        )

    return ppo_model


def main(args):
    """train the agent """
    assert args.save_frequency == -1 \
        or 0 <= args.save_frequency <= args.total_timesteps \
        or args.save_frequency >= args.batch_steps * args.num_cpus

    print("Running baseline3 PPO using args:")
    pprint(args)

    ppo_model = init_model(args)
    run_model(ppo_model, args)

    reset_num_timesteps = args.load_model == ""
    training_complete = args.total_timesteps <= 0
    while not training_complete:
        train_model(ppo_model, args, reset_num_timesteps)
        reset_num_timesteps = False
        run_model(ppo_model, args, wait_for_user=True)
        training_complete = not continue_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--independence",  type=float, default=0.0,
                        help="driver independence (default=0.0)")
    parser.add_argument("-d", "--driver_type", type=str,
                        default="GuidedIDMDriverVehicle",
                        help="driver type (default='GuidedIDMDriverVehicle'")
    parser.add_argument("-t", "--total_timesteps", type=int, default=8192,
                        help=("training timesteps (default=8192). Set this"
                              " to 0 to eval model only"))
    parser.add_argument("-b", "--batch_steps", type=int, default=256,
                        help="num steps per update (default=512)")
    parser.add_argument("-nc", "--num_cpus", type=int, default=1,
                        help="num cpus to use (default=1)")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Verbosity level (default=1)")
    parser.add_argument("-f", "--simulation_frequency", type=int, default=15,
                        help="env simulation frequency (default=15)")
    parser.add_argument("-ss", "--stack_size", type=int, default=1,
                        help="number of observations to stack (default=1)")
    parser.add_argument("-s", "--save_frequency", type=int, default=-1,
                        help=("model save frequency (in timesteps) (default="
                              "total_timesteps). If set must be 0 <= s <= "
                              "total_timesteps. 0 means no save. "))
    parser.add_argument("-l", "--load_model", type=str, default="",
                        help="path to model to load (default=None)")
    main(parser.parse_args())
