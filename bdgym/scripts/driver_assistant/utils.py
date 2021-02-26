"""Utility functions for running tests on BDGym autopilot env """
import os
import os.path as osp
import time
from pprint import pprint
from typing import Tuple, Union
from collections import namedtuple
from argparse import ArgumentParser, Namespace

import numpy as np

from bdgym.envs.driver_assistant.driver_types import \
    get_driver_config, AVAILABLE_DRIVER_TYPES
from bdgym.envs.driver_assistant.fixed_driver_env import \
    FixedDriverDriverAssistantEnv
import bdgym.envs.driver_assistant.policy as policy


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)

NUM_CPUS = max(1, len(os.sched_getaffinity(0)) - 4)


Result = namedtuple(
    "Result",
    [
        "assistant",
        "driver",
        "independence",
        "episodes",
        "seed",
        "return_mean",
        "return_std",
        "steps_mean",
        "steps_std",
        "collision_prob",
        "time_mean",
        "time_std",
        "deception_mean",
        "deception_std"
    ]
)


RunArgs = namedtuple(
    "RunArgs",
    [
        "independence",
        "driver_type",
        "assistant_type",
        "num_episodes",
        "seed",
        "render",
        "normalize_obs",
        "verbose",
        "manual",
        "discrete",
        "force_independent",
        "time_delay"
    ]
)


def display_result(result: Result):
    """Display result in stdout """
    print(f"\n{'-'*60}")
    print(f"Run results for {result.episodes} episodes")
    print(f"Assistant={result.assistant}")
    print(f"Driver={result.driver}")
    print(
        f"Mean return = {result.return_mean:.3f} "
        f"+/- {result.return_std:.3f}"
    )
    print(f"Mean steps = {result.steps_mean:.3f} +/- {result.steps_std:.3f}")
    print(f"Collision prob = {result.collision_prob:.3f}")
    print(f"Mean time = {result.time_mean:.3f} +/- {result.time_std:.3f}")
    print(
        f"Mean deception = {result.deception_mean:.3f} "
        f"+/- {result.deception_std:.3f}"
    )
    print(f"{'-'*60}\n")


def test_parser() -> ArgumentParser:
    """Initialize command line parser for test scripts """
    parser = ArgumentParser()
    parser.add_argument("-i", "--independence", type=float, default=0.0,
                        help="driver independence (default=0.0)")
    parser.add_argument("-d", "--driver_type", type=str, default="standard",
                        help=(f"Can be one of {AVAILABLE_DRIVER_TYPES} or "
                              "'changing' (default='standard')."))
    parser.add_argument("-a", "--assistant_type", type=str,
                        default="standard",
                        help=(
                            "The assistant type, can be same as driver_types"
                            " or 'random' (default='standard')"
                        ))
    parser.add_argument("-e", "--num_episodes", type=int, default=1,
                        help="Number of episodes (default=100)")
    parser.add_argument("-l", "--episode_length", type=int, default=-1,
                        help=(
                            "Episode length (is scaled by simulation "
                            "frequency (default=use default env length)"
                        ))
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="random seed (default=None)")
    parser.add_argument("-r", "--render", action="store_true",
                        help="Render episodes")
    parser.add_argument("-no", "--normalize_obs", action="store_true",
                        help="Normalize Observations to be in [-1, 1]")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbosity mode")
    parser.add_argument("-m", "--manual", action="store_true",
                        help="Manual control mode")
    parser.add_argument("-dc", "--discrete", action="store_true",
                        help="Use Discrete Assistant Action")
    parser.add_argument("-fi", "--force_independent", action="store_true",
                        help=("Ensure driver is independent "
                              "(mainly used for 'changing' driver type)"))
    parser.add_argument("-t", "--time_delay", type=float, default=0.0,
                        help="inter-step time delay (default=0.0)")
    return parser


def parse_parser(parser: ArgumentParser) -> Namespace:
    """Parse command line parser, returning parsed arguments

    Also prints message with arg settings
    """
    args = parser.parse_args()

    if args.verbose:
        print("\nUsing command line arguments:")
        pprint(args)
    return args


def get_configured_env(args: Union[Namespace, RunArgs], seed: int = None):
    """Get the configured env """
    env = FixedDriverDriverAssistantEnv()

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)
    else:
        env.seed(args.seed)
        np.random.seed(args.seed)

    if args.driver_type == "changing":
        driver_policy = {
            "type": "ChangingGuidedIDMDriverPolicy",
            "force_independent": args.force_independent
        }
    elif args.driver_type == 'random':
        driver_policy = {
            "type": "RandomDriverPolicy"
        }
    else:
        driver_policy = {
            "type": "GuidedIDMDriverPolicy",
            "independence": args.independence,
            **get_driver_config(args.driver_type)
        }

    config = {
        "driver_policy": driver_policy,
        "manual_control": False
    }

    action_config = env.config["action"]
    action_config["clip"] = False

    if args.manual:
        config["manual_control"] = True
        action_config["assistant"]["type"] = "AssistantDiscreteActionSpace"

    if args.discrete:
        action_config["assistant"]["type"] = "AssistantDiscreteActionSpace"

    config["action"] = action_config

    if not args.normalize_obs:
        config["observation"] = {
            "type": "StackedKinematicObservation",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": False,
            "absolute": True,
            "order": "sorted",
            "stack_size": 1
        }

    env.configure(config)
    env.reset()

    if args.verbose:
        print("\nFull Env Config:")
        pprint(env.config)
        print()
    return env


def init_assistant(args: Union[Namespace, RunArgs],
                   env: FixedDriverDriverAssistantEnv
                   ) -> policy.IDMAssistantPolicy:
    """Initialize assistant """
    kwargs = {}
    if args.assistant_type.lower() == 'random':
        if args.discrete:
            assistant_class = policy.RandomDiscreteAssistantPolicy
        else:
            assistant_class = policy.RandomDriverPolicy
    else:
        kwargs.update(get_driver_config(args.assistant_type))
        assistant_class = policy.IDMAssistantPolicy

    kwargs["normalize"] = True
    kwargs["action_ranges"] = \
        env.config["action"]["assistant"]["features_range"]
    return assistant_class.create_from(env.vehicle, **kwargs)


def run_episode(args: Union[Namespace, RunArgs],
                env: FixedDriverDriverAssistantEnv
                ) -> Tuple[float, int, bool, float]:
    """Run the env for a single episode"""
    obs = env.reset()

    assistant = init_assistant(args, env)

    obs = env.reset()
    if args.render:
        env.render()

    done = False
    total_return = 0.0
    steps = 0
    while not done:
        action = assistant.get_action(obs, env.delta_time)
        obs, reward, done, _ = env.step(action)
        total_return += reward
        steps += 1

        if args.render:
            env.render()

        time.sleep(args.time_delay)

    mean_deception = np.mean(env.assistant_deception)
    collision = steps < env.config["duration"]
    return total_return, steps, collision, mean_deception


def run(args: Union[Namespace, RunArgs]) -> Result:
    """Run FixedDriverDriverAssistantEnv """
    env = get_configured_env(args)

    display_freq = max(args.num_episodes // 10, 1)

    ep_returns = []
    ep_collisions = []
    ep_steps = []
    ep_times = []
    ep_deceptions = []
    for e in range(args.num_episodes):
        start_time = time.time()
        total_return, steps, collision, deception = run_episode(args, env)
        ep_times.append(time.time() - start_time)
        ep_returns.append(total_return)
        ep_steps.append(steps)
        ep_collisions.append(collision)
        ep_deceptions.append(deception)

        if args.verbose and e > 0 and e % display_freq == 0:
            print(
                f"Episode {e} complete: "
                f"return={total_return:.3f} steps={steps} "
                f"collision={collision} time={ep_times[-1]:.3f} "
                f"deception={deception:.3f}"
            )

    result = Result(
        assistant=args.assistant_type,
        driver=args.driver_type,
        independence=args.independence,
        episodes=args.num_episodes,
        seed=args.seed,
        return_mean=np.mean(ep_returns),
        return_std=np.std(ep_returns),
        steps_mean=np.mean(ep_steps),
        steps_std=np.std(ep_steps),
        collision_prob=np.mean(ep_collisions),
        time_mean=np.mean(ep_times),
        time_std=np.std(ep_times),
        deception_mean=np.mean(ep_deceptions),
        deception_std=np.std(ep_deceptions)
    )

    if args.verbose:
        display_result(result)

    return result
