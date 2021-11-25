"""Key board agent for the Benevolent Deception Gym """
import time
from typing import Tuple, Optional

import gym
import numpy as np

import bdgym
from bdgym.envs.exercise_assistant.resources import display_keybindings
from bdgym.envs.driver_assistant.manual_control import AssistantEventHandler
from bdgym.envs.exercise_assistant.policy import (
    ManualAssistantPolicy, ManualDiscreteAssistantPolicy
)


LINE_BREAK = "-" * 60


def run_exercise_assistant(env_name: str) -> Tuple[float, int, bool, float]:
    """Run Keyboard agent on the Exercise Assistant Environment.

    Parameters
    ----------
    env_name : str
        the name of Exercise-Assistant Env to run
    """
    env = gym.make(env_name)

    if "Continuous" in env_name:
        assistant_policy = ManualAssistantPolicy(False, False)
    else:
        assistant_policy = ManualDiscreteAssistantPolicy(False, True)
        answer = input(
            "Would you like to view the keybindings for the environment "
            "[y]/N? "
        )
        if answer.lower() != "n":
            display_keybindings()
            input("Press ENTER to start environment.")

    obs = env.reset()
    env.render('human')

    total_return = 0.0
    steps = 0
    done = False
    start_time = time.time()
    while not done:
        print(LINE_BREAK)
        print(f"ASSISTANT Step = {steps}")
        print(LINE_BREAK)
        action = assistant_policy.get_action(obs)
        obs, rew, done, _ = env.step(action)
        total_return += rew

        print(f"\nReward: {rew}")
        print(f"Done: {done}")
        steps += 1

        env.render('human')

    deception_mean = np.mean(env.assistant_deception)
    time_taken = time.time() - start_time

    print(LINE_BREAK)
    print("EPISODE COMPLETE")
    print(LINE_BREAK)
    print(
        f"return = {total_return:.3f}\n"
        f"steps = {steps}\n"
        f"overexerted = {env.athlete_overexerted()}\n"
        f"mean deception = {deception_mean:.3f}\n"
        f"time = {time_taken:.3f} seconds"
    )

    return total_return, steps, env.athlete_overexerted(), deception_mean


def run_driver_assistant(env_name: str,
                         seed: Optional[int] = None
                         ) -> Tuple[float, int, bool, float]:
    """Run Keyboard agent on the Exercise Assistant Environment.

    Parameters
    ----------
    env_name : str
        the name of Exercise-Assistant Env to run
    seed : int, optional
        random seed (default=None)
    """
    env = gym.make(env_name)
    env.config["manual_control"] = True

    env.seed(seed)
    env.reset()
    env.render('human')

    total_return = 0.0
    steps = 0
    done = False
    start_time = time.time()
    while not done:
        action = AssistantEventHandler.get_discrete_action(env)
        _, rew, done, _ = env.step(action)
        total_return += rew
        steps += 1

        env.render('human')

    deception_mean = np.mean(env.assistant_deception)
    crashed = steps < env.config["duration"]
    time_taken = time.time() - start_time

    print(LINE_BREAK)
    print("EPISODE COMPLETE")
    print(LINE_BREAK)
    print(
        f"return = {total_return:.3f}\n"
        f"steps = {steps}\n"
        f"crashed = {crashed}\n"
        f"mean deception = {deception_mean:.3f}\n"
        f"time = {time_taken:.3f} seconds"
    )

    return total_return, steps, crashed, deception_mean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="Name of BDGym environment to run")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG Seed (default=None)")
    args = parser.parse_args()

    assert args.env_name in bdgym.envs.BDGYM_ENVS, \
        (f"'{args.env_name}' is not a valid BDGym environment name.\n"
         f"The available environments are {bdgym.envs.BDGYM_ENVS}")

    if args.env_name in bdgym.envs.ALL_EXERCISE_ASSISTANT_GYM_ENVS:
        run_exercise_assistant(args.env_name)
    elif args.env_name in bdgym.envs.ALL_DRIVER_ASSISTANT_GYM_ENVS:
        run_driver_assistant(args.env_name, args.seed)
