"""Utility functions and classes for training and running Deep RL models """
import os.path as osp
from typing import List
from argparse import ArgumentParser

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed, logger
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


# Default name for saved best model
BEST_MODEL_FILE_NAME = "best_model.zip"


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
        return get_configured_env_fn(seed)
    if eval_env:
        num_envs = 1
    else:
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


def load_model(model_cls, load_path: str, env: gym.Env) -> PPO:
    """Load a model from save file """
    return model_cls.load(load_path, env=env)


def load_best_model(model_cls,
                    save_dir: str,
                    env: gym.Env) -> BaseAlgorithm:
    """Load best model (if it exists) from save dir """
    load_path = osp.join(save_dir, BEST_MODEL_FILE_NAME)
    return load_model(model_cls, load_path, env)


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


def get_training_callbacks(eval_env: gym.Env,
                           result_dir: str,
                           save_frequency: int = 0,
                           save_best: bool = False,
                           eval_freq: int = 0,
                           n_eval_episodes: int = 10):
    """Get Callbacks to use during training """
    callback_list = []
    if save_frequency != 0:
        checkpoint_callback = CheckpointCallback(
            save_frequency, save_path=result_dir
        )
        callback_list.append(checkpoint_callback)

    if save_best:
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=result_dir,
            best_model_save_path=result_dir
        )
        callback_list.append(eval_callback)

    return callback_list


def train_model(model: BaseAlgorithm,
                total_timesteps: int,
                eval_env: gym.Env,
                save_frequency: int,
                eval_freq: int,
                reset_num_timesteps: bool,
                log_name: str,
                save_best: bool,
                result_dir: str,
                n_eval_episodes: int = 10,
                **learn_kwargs):
    """Run a single training cycle of the model """
    print(f"Training model for {total_timesteps} steps")
    if save_frequency <= 0:
        save_frequency = total_timesteps

    callback_list = get_training_callbacks(
        eval_env,
        result_dir,
        save_frequency,
        save_best,
        eval_freq,
        n_eval_episodes,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=log_name,
        reset_num_timesteps=reset_num_timesteps,
        **learn_kwargs
    )


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
                        help="num steps per update (default=256)")
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
    parser.add_argument("-sb", "--save_best", action="store_true",
                        help="Save the best evaluated model")
    parser.add_argument("-l", "--load_model", type=str, default="",
                        help="path to model to load (default=None)")
    return parser
