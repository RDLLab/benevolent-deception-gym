"""Utility functions and classes for training and running Deep RL models """
from typing import List
from argparse import ArgumentParser

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed, logger


def make_env(get_configured_env_fn, rank: int, seed: int = 0):
    """Utility function for multiprocessed env. """
    def _init():
        env = get_configured_env_fn(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_env(get_configured_env_fn,
            eval_env: bool = False,
            num_cpus: int = 1,
            seed: int = 0) -> gym.Env:
    """Get the agent env """
    if num_cpus == 1:
        print("Using single environment")
        return get_configured_env_fn(seed)
    if eval_env:
        num_envs = 1
    else:
        print(f"Running {num_cpus} envs in parallel")
        num_envs = num_cpus

    env = SubprocVecEnv(
        [make_env(get_configured_env_fn, i, seed) for i in range(num_envs)]
    )

    return env


def save_model(model: BaseAlgorithm):
    """Save model """
    print("Saving model")
    model_save_path = logger.get_dir()
    model.save(model_save_path)
    print(f"Model saved to = {model_save_path}")


def get_train_epoch_lengths(save_frequency: int,
                            total_timesteps: int) -> List[int]:
    """Get the lengths of learning epochs. Here One epoch is the number
    of timesteps between when the model is saved.
    """
    if save_frequency <= 0:
        epoch_lengths = [total_timesteps]
    else:
        epoch_lengths = []
        remaining_timesteps = total_timesteps
        while remaining_timesteps > 0:
            epoch_lengths.append(min(remaining_timesteps, save_frequency))
            remaining_timesteps -= save_frequency
    return epoch_lengths


def train_model(model: BaseAlgorithm,
                total_timesteps: int,
                eval_env: gym.Env,
                save_frequency: int,
                eval_freq: int,
                reset_num_timesteps: bool,
                log_name: str,
                n_eval_episodes: int = 10,
                **learn_kwargs):
    """Run a single training cycle of the model """
    print(f"Training model for {total_timesteps} steps")
    epoch_lengths = get_train_epoch_lengths(save_frequency, total_timesteps)
    print(f"With epochs of sizes: {epoch_lengths}")

    for i, t in enumerate(epoch_lengths):
        print(f"Starting epoch {i+1} of {len(epoch_lengths)}")
        model.learn(
            total_timesteps=t,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=log_name,
            reset_num_timesteps=reset_num_timesteps,
            **learn_kwargs
        )
        reset_num_timesteps = False
        if save_frequency != 0:
            save_model(model)


def run_model(model: BaseAlgorithm,
              env: gym.Env,
              verbosity: int = 0,
              wait_for_user: bool = False):
    """Run model on env """
    print("Running episode of the model")
    if wait_for_user:
        input("Press Enter to run episode")

    obs = env.reset()
    if verbosity > 1:
        print(f"\nObservation=\n{np.array_str(obs, precision=3)}")
    done = False
    total_return = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if verbosity > 1:
            print(
                f"\nAction={np.array_str(action, precision=3)}"
                f"\nObservation=\n{np.array_str(obs, precision=3)}"
                f"\nReward={reward:.3f}"
                f"\nDone={done}"
            )
        total_return += reward
        env.render()
        if verbosity > 2:
            input()
    print(f"Return = {total_return}")
    env.close()


def init_ppo_model(env: gym.Env,
                   batch_steps: int,
                   verbosity: int,
                   result_dir: str,
                   policy: str = 'MlpPolicy',
                   **ppo_kwargs) -> PPO:
    """Initialize a new PPO model """
    ppo_model = PPO(
        policy,
        env,
        n_steps=batch_steps,
        verbose=verbosity,
        tensorboard_log=result_dir,
        **ppo_kwargs
    )
    return ppo_model


def load_ppo_model(load_path: str, env: gym.Env) -> PPO:
    """Load a PPO model from save file """
    return PPO.load(load_path, env=env)


def continue_training() -> bool:
    """Check whether to continue training or not """
    answer = input("Continue Training [y/n]?: ")
    if answer.lower() != "y":
        print("Ending training")
        return False
    print("Running another round of learning")
    return True


def get_ppo_argparse(parser: ArgumentParser = None) -> ArgumentParser:
    """Get or add argument parser for PPO """
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("-t", "--total_timesteps", type=int, default=8192,
                        help=("training timesteps (default=8192). Set this"
                              " to 0 to eval model only"))
    parser.add_argument("-b", "--batch_steps", type=int, default=256,
                        help="num steps per update (default=512)")
    parser.add_argument("-sd", "--seed", type=int, default=0,
                        help="seed (default=0)")
    parser.add_argument("-nc", "--num_cpus", type=int, default=1,
                        help="num cpus to use (default=1)")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (default=1)")
    parser.add_argument("-s", "--save_frequency", type=int, default=-1,
                        help=("model save frequency (in timesteps) (default="
                              "total_timesteps). If set must be 0 <= s <= "
                              "total_timesteps. 0 means no save. "))
    parser.add_argument("-l", "--load_model", type=str, default="",
                        help="path to model to load (default=None)")
    return parser
