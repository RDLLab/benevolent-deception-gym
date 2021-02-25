"""Utility functions and classes for running the Exercise Assistant Env """
import os
import os.path as osp
import time
import datetime
from pprint import pprint
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from typing import Tuple, Union, Callable, Dict, List

import numpy as np

import bdgym.envs.exercise_assistant as ea_env
import bdgym.envs.exercise_assistant.policy as policy


RESULTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "results"
)

NUM_CPUS = max(1, len(os.sched_getaffinity(0)) - 4)


LINE_BREAK = "-" * 60


Result = namedtuple(
    "Result",
    [
        "assistant",
        "athlete",
        "independence",
        "perception_influence",
        "episodes",
        "seed",
        "return_mean",
        "return_std",
        "steps_mean",
        "steps_std",
        "overexertion_prob",
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
        "perception_influence",
        "fixed_athlete_policy",
        "fixed_assistant_policy",
        "num_episodes",
        "seed",
        "render",
        "no_athlete_render",
        "no_assistant_render",
        "verbose",
        "manual",
        "discrete"
    ]
)


def display_result(result: Result):
    """Display result in stdout """
    print(f"\n{'-'*60}")
    print(f"Run results for {result.episodes} episodes")
    print(f"Assistant={result.assistant}")
    print(f"Athlete={result.athlete}")
    print(
        f"Mean return = {result.return_mean:.3f} "
        f"+/- {result.return_std:.3f}"
    )
    print(f"Mean steps = {result.steps_mean:.3f} +/- {result.steps_std:.3f}")
    print(f"Overexertion prob = {result.overexertion_prob:.3f}")
    print(f"Mean time = {result.time_mean:.3f} +/- {result.time_std:.3f}")
    print(
        f"Mean deception = {result.deception_mean:.3f} "
        f"+/- {result.deception_std:.3f}"
    )
    print(f"{'-'*60}\n")


def argument_parser() -> ArgumentParser:
    """Initialize command line parser for test scripts """
    parser = ArgumentParser()
    parser.add_argument("-i", "--independence", type=float, default=0.0,
                        help="athlete independence (default=0.0)")
    parser.add_argument("-pi", "--perception_influence",
                        type=float, default=0.5,
                        help="athlete perception influence (default=0.5)")
    parser.add_argument("-fatp", "--fixed_athlete_policy", type=str,
                        default='',
                        help=(
                            "Use fixed athlete policy from: "
                            f"{list(policy.ATHLETE_POLICIES)}. Or use '' for "
                            "no fixed athlete policy env (default='')"
                        ))
    parser.add_argument("-fasp", "--fixed_assistant_policy", type=str,
                        default="",
                        help=(
                            "Use fixed assistant policy from: "
                            f"{list(policy.ASSISTANT_POLICIES)}. Or use '' "
                            "for no fixed assistant policy env (default='')"
                        ))
    parser.add_argument("-e", "--num_episodes", type=int, default=1,
                        help="Number of episodes (default=100)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="random seed (default=None)")
    parser.add_argument("-r", "--render", type=str, default="",
                        help="Render mode (default=''=no render)")
    parser.add_argument("--no_athlete_render", action="store_true",
                        help="Don't render athlete info")
    parser.add_argument("--no_assistant_render", action="store_true",
                        help="Don't render assistant info")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbosity mode")
    parser.add_argument("-m", "--manual", action="store_true",
                        help="Manual control mode")
    parser.add_argument("-d", "--discrete", action="store_true",
                        help="Use discrete Assistant actions")
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


def init_athlete_policy(args: Union[Namespace, RunArgs]
                        ) -> policy.AthletePolicy:
    """Initialize the athlete policy given args """
    assert args.fixed_athlete_policy in policy.ATHLETE_POLICIES
    athlete_policy_cls = policy.ATHLETE_POLICIES[args.fixed_athlete_policy]

    if athlete_policy_cls == policy.WeightedAthletePolicy:
        return athlete_policy_cls(
            perception_influence=args.perception_influence,
            independence=args.independence
        )
    return athlete_policy_cls()


def init_assistant_policy(args: Union[Namespace, RunArgs]
                          ) -> policy.AssistantPolicy:
    """Initialize the assistant policy for given args """
    if args.fixed_assistant_policy == '':
        if args.discrete:
            return policy.ManualDiscreteAssistantPolicy()
        return policy.ManualAssistantPolicy()

    assert args.fixed_assistant_policy in policy.ASSISTANT_POLICIES
    assistant_policy_cls = policy.ASSISTANT_POLICIES[
        args.fixed_assistant_policy
    ]
    return assistant_policy_cls()


def get_configured_env(args: Union[Namespace, RunArgs], seed: int = None):
    """Get the configured env """
    use_fixed_athlete = args.fixed_athlete_policy != ''

    env_kwargs = {
        "render_assistant_info": not args.no_assistant_render,
        "render_athlete_info": not args.no_athlete_render
    }

    if use_fixed_athlete:
        athlete_policy = init_athlete_policy(args)
        if args.discrete:
            env = ea_env.DiscreteFixedAthleteExerciseAssistantEnv(
                athlete_policy=athlete_policy, **env_kwargs
            )
        else:
            env = ea_env.FixedAthleteExerciseAssistantEnv(
                athlete_policy=athlete_policy, **env_kwargs
            )
    else:
        if args.discrete:
            env = ea_env.DiscreteExerciseAssistantEnv()
        else:
            env = ea_env.ExerciseAssistantEnv()

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)
    else:
        env.seed(args.seed)
        np.random.seed(args.seed)

    return env


def run_fixed_athlete_episode(args: Union[Namespace, RunArgs],
                              env: ea_env.FixedAthleteExerciseAssistantEnv,
                              assistant_policy: policy.AssistantPolicy
                              ) -> Tuple[float, int, bool, float]:
    """Run fixed athlete policy env for a single episode"""
    obs = env.reset()
    if args.render != '':
        env.render(args.render)

    done = False
    total_return = 0.0
    steps = 0
    while not done:
        action = assistant_policy.get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_return += reward
        steps += 1

        if args.render != '':
            env.render(args.render)

    deception_mean = np.mean(env.assistant_deception)
    return total_return, steps, env.athlete_overexerted(), deception_mean


def run_manual_fixed_athlete_episode(
        args: Union[Namespace, RunArgs],
        env: ea_env.FixedAthleteExerciseAssistantEnv
) -> Tuple[float, int, bool, float]:
    """Run a single episode """
    obs = env.reset()

    if not args.discrete:
        assistant_policy = policy.ManualAssistantPolicy(True)
    else:
        assistant_policy = policy.ManualDiscreteAssistantPolicy(True)

    if args.render != '':
        env.render(args.render)

    total_return = 0.0
    steps = 0
    done = False
    while not done:
        print(LINE_BREAK)
        print(f"ASSISTANT Step = {steps}")
        print(LINE_BREAK)
        action = assistant_policy.get_action(obs)
        obs, rew, done, _ = env.step(action)
        total_return += rew

        if args.verbose:
            print(f"\nReward: {rew}")
            print(f"Done: {done}")
        steps += 1

        if args.render != '':
            env.render(args.render)

    deception_mean = np.mean(env.assistant_deception)

    return total_return, steps, env.athlete_overexerted(), deception_mean


def run_manual_episode(args: Union[Namespace, RunArgs],
                       env: ea_env.ExerciseAssistantEnv
                       ) -> Tuple[float, int, bool, float]:
    """Run a single episode """
    ea_obs = env.reset()

    if not args.discrete:
        assistant_policy = policy.ManualAssistantPolicy(True)
    else:
        assistant_policy = policy.ManualDiscreteAssistantPolicy(True)

    athlete_policy = policy.ManualAthletePolicy(True)

    if args.render:
        env.render()

    total_return = 0.0
    steps = 0
    done = False
    while not done:
        print(LINE_BREAK)
        print(f"ASSISTANT Step = {steps}")
        print(LINE_BREAK)
        ea_action = assistant_policy.get_action(ea_obs)
        at_obs, rew, done, _ = env.step(ea_action)

        print(LINE_BREAK)
        print(f"ATHLETE Step = {steps}")
        print(LINE_BREAK)
        at_action = athlete_policy.get_action(at_obs)
        ea_obs, rew, done, _ = env.step(at_action)
        if args.verbose:
            print(f"\nReward: {rew}")
            print(f"Done: {done}")
        total_return += rew

        steps += 1

        if args.render:
            env.render()

    deception_mean = np.mean(env.assistant_deception)
    return total_return, steps, env.athlete_overexerted(), deception_mean


def get_run_fn(args: Union[Namespace, RunArgs]) -> Tuple[Callable, Dict]:
    """Get the episode run function and function kwargs """
    use_fixed_athlete = args.fixed_athlete_policy != ''
    use_fixed_assistant = args.fixed_assistant_policy != ''

    run_kwargs = {}
    if not use_fixed_athlete and not use_fixed_assistant:
        run_fn = run_manual_episode
    elif use_fixed_athlete and not use_fixed_assistant:
        run_fn = run_manual_fixed_athlete_episode
    elif use_fixed_athlete and use_fixed_assistant:
        run_fn = run_fixed_athlete_episode
        run_kwargs["assistant_policy"] = init_assistant_policy(args)
    else:
        raise ValueError(
            "Currently do not support running a fixed assistant policy with "
            "manual athlete control"
        )
    return run_fn, run_kwargs


def run(args: Union[Namespace, RunArgs]) -> Result:
    """Run Fixed Athlete Policy Exercise Assistant Env """
    env = get_configured_env(args)
    run_fn, run_kwargs = get_run_fn(args)

    display_freq = max(args.num_episodes // 10, 1)

    ep_returns = []
    ep_overexertions = []
    ep_steps = []
    ep_times = []
    ep_deception = []
    for e in range(args.num_episodes):
        start_time = time.time()
        total_return, total_steps, overexerted, deception = run_fn(
            args, env, **run_kwargs
        )
        ep_times.append(time.time() - start_time)
        ep_returns.append(total_return)
        ep_steps.append(total_steps)
        ep_overexertions.append(int(overexerted))
        ep_deception.append(deception)

        if args.verbose and e > 0 and e % display_freq == 0:
            print(
                f"Episode {e} complete: "
                f"return={total_return:.3f} steps={total_steps} "
                f"overexerted={overexerted} time={ep_times[-1]:.3f} "
                f"deception={deception:.3f}"
            )

    result = Result(
        assistant=args.fixed_assistant_policy,
        athlete=args.fixed_athlete_policy,
        independence=args.independence,
        perception_influence=args.perception_influence,
        episodes=args.num_episodes,
        seed=args.seed,
        return_mean=np.mean(ep_returns),
        return_std=np.std(ep_returns),
        steps_mean=np.mean(ep_steps),
        steps_std=np.std(ep_steps),
        overexertion_prob=np.mean(ep_overexertions),
        time_mean=np.mean(ep_times),
        time_std=np.std(ep_times),
        deception_mean=np.mean(ep_deception),
        deception_std=np.std(ep_deception)
    )

    if args.verbose:
        display_result(result)

    return result


def save_results(all_results: List[Result],
                 filename: str,
                 include_timestamp: bool = True):
    """Save all results to file """
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        full_filename = osp.join(RESULTS_DIR, f"{filename}_{timestamp}.csv")
    else:
        full_filename = osp.join(RESULTS_DIR, f"{filename}_{timestamp}.csv")

    print(f"\nSaving results to: {full_filename}")

    with open(full_filename, "w") as fout:
        headers = all_results[0]._fields
        fout.write("\t".join(headers) + "\n")
        for result in all_results:
            row = [str(v) for v in result._asdict().values()]
            fout.write("\t".join(row) + "\n")


def append_result_to_file(results: Union[List[Result], Result],
                          filepath: str,
                          add_header: bool = True):
    """Append result to file

    Will add header if add_header is True and the file at filepath doesn't
    exist
    """
    if not isinstance(results, list):
        results = [results]

    if not osp.isfile(filepath) and add_header:
        with open(filepath, "w") as fout:
            headers = results[0]._fields
            fout.write("\t".join(headers) + "\n")

    with open(filepath, "a") as fout:
        for result in results:
            row = []
            for v in result._asdict().values():
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            fout.write("\t".join(row) + "\n")
