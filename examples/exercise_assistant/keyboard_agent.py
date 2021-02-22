"""A keyboard agent for the Percieved Effort Environment """
import bdgym.envs.utils as utils
import bdgym.envs.exercise_assistant.policy as policy
from bdgym.envs.exercise_assistant.observation import \
    assistant_obs_str, athlete_obs_str
from bdgym.envs.exercise_assistant.action import \
    AthleteAction, DiscreteAssistantAction, assistant_action_str
from bdgym.envs import \
    ExerciseAssistantEnv, DiscreteExerciseAssistantEnv, \
    FixedAthleteExerciseAssistantEnv, DiscreteFixedAthleteExerciseAssistantEnv

import numpy as np


LINE_BREAK = "-" * 60


def display_init_obs(ea_obs=None, at_obs=None):
    """Display initial observation """
    print(LINE_BREAK)
    if ea_obs is not None:
        print("ASSISTANT")
        obs_str = assistant_obs_str(ea_obs)
        print(f"\tObservation: {obs_str}")
        print()
    if at_obs is not None:
        print("ATHLETE")
        obs_str = athlete_obs_str(at_obs)
        print(f"\tObservation: {obs_str}")


def display_assistant_step(args, steps, action, obs, rew, done):
    """Display a single assistant step """
    if args.fixed_athlete != '':
        obs_str = f"\tObservation: {assistant_obs_str(obs)}"
    else:
        obs_str = f"\tAthlete Observation: {athlete_obs_str(obs)}"

    print(LINE_BREAK)
    print(f"ASSISTANT Step = {steps}")
    print(LINE_BREAK)
    print(f"\tAction: {assistant_action_str(action)}")
    print(obs_str)
    print(f"\tReward: {rew}")
    print(f"\tDone: {done}")
    print()


def display_athlete_step(steps, action, obs, rew, done):
    """Display a single athlete step """
    print(LINE_BREAK)
    print(f"ATHLETE Step = {steps}")
    print(LINE_BREAK)
    print(f"\tAction: {str(AthleteAction(action))}")
    print(f"\tAssistant Observation: {assistant_obs_str(obs)}")
    print(f"\tReward: {rew}")
    print(f"\tDone: {done}")
    print()


def display_episode_end(args, steps, total_return, final_state):
    """Display end of episode message """
    if args.fixed_athlete != '':
        return_str = f"Return = {total_return}"
    else:
        return_str = (
            f"Assistant return = {total_return[0]}\n"
            f"Athlete return = {total_return[1]}"
        )

    print(LINE_BREAK)
    print("Episode Complete")
    print(LINE_BREAK)
    print(f"Steps = {steps}")
    print(f"Energy remaining = {final_state[0]:.4f}")
    print(f"Sets Done = {final_state[1]:.4f}")
    print(return_str)
    print(LINE_BREAK)


def get_discrete_action(agent_name, action_space_cls):
    """Get discrete action selection from user """
    print(LINE_BREAK)
    print(f"Select {agent_name} Action from:")

    output = ["Actions:"]
    for i in range(len(action_space_cls)):
        output.append(f"{i}={str(action_space_cls(i))}")
    print(" ".join(output))

    while True:
        try:
            idx = int(input("Choose action number: "))
            action = action_space_cls(idx)
            print(f"Performing: {str(action)}")
            return action
        except ValueError:
            print("Invalid choice. Try again.")


def get_athlete_action():
    """Get action from user """
    return get_discrete_action("Athlete", AthleteAction)


def get_assistant_action(args):
    """Get action from user """
    print(LINE_BREAK)
    if args.discrete:
        return get_discrete_action("Assistant", DiscreteAssistantAction)

    signal = 0.0
    rcmd = 0.0
    print("Select Assistant Action (energy to report in [0.0, 1.0]:")
    while True:
        try:
            signal = float(input("Energy: "))
            assert 0.0 <= signal <= 1.0
            break
        except (ValueError, AssertionError):
            print("Invalid choice. Try again.")

    print("Select Assistant recommended Action (in [0.0, 1.0]:")
    while True:
        try:
            rcmd = float(input("Recommendation: "))
            assert 0.0 <= rcmd <= 1.0
            break
        except (ValueError, AssertionError):
            print("Invalid choice. Try again.")

    action = np.array([signal, rcmd])
    print(f"Performing: {np.array_str(action, precision=4)}")
    return utils.lmap_array(action, [0.0, 1.0], [-1.0, 1.0])


def run_episode(env, args):
    """Run a single episode """
    ea_obs, at_obs = env.reset()
    display_init_obs(ea_obs, at_obs)

    if args.render:
        env.render()

    total_return = np.zeros(2)
    steps = 0
    done = False
    while not done:
        ea_action = get_assistant_action(args)
        at_obs, rew, done, _ = env.step(ea_action)
        display_assistant_step(args, steps, ea_action, at_obs, rew, done)
        total_return[0] += rew

        at_action = get_athlete_action()
        ea_obs, rew, done, _ = env.step(at_action)
        display_athlete_step(steps, at_action, ea_obs, rew, done)
        total_return[1] += rew

        steps += 1

        if args.render:
            env.render()

    display_episode_end(args, steps, total_return, env.state)
    return steps, total_return


def run_fixed_athlete_episode(env, args):
    """Run a single episode """
    obs = env.reset()
    display_init_obs(obs)

    if args.render:
        env.render()

    total_return = 0
    steps = 0
    done = False
    while not done:
        action = get_assistant_action(args)
        obs, rew, done, _ = env.step(action)
        display_assistant_step(args, steps, action, obs, rew, done)
        total_return += rew
        steps += 1

        if args.render:
            env.render()

    display_episode_end(args, steps, total_return, env.state)
    return steps, total_return


def load_athlete_policy(args):
    """Initialize the athlete policy """
    policy_name = args.fixed_athlete.lower()
    if policy_name in policy.ATHLETE_POLICIES:
        return policy.ATHLETE_POLICIES[policy_name]()
    raise ValueError(
        f"Invalid value for 'fixed_athlete' argument: '{args.fixed_athlete}'. "
        f"Must be one of: {list(policy.ATHLETE_POLICIES)} or ''"
    )


def get_env(args):
    """Initialize the environment """
    if args.fixed_athlete == '':
        if args.discrete:
            return DiscreteExerciseAssistantEnv()
        return ExerciseAssistantEnv()

    athlete_policy = load_athlete_policy(args)
    if args.discrete:
        return DiscreteFixedAthleteExerciseAssistantEnv(athlete_policy)
    return FixedAthleteExerciseAssistantEnv(athlete_policy)


def main(args):
    """Run keyboard agent for Exercise Assistant Environment """
    env = get_env(args)
    if args.fixed_athlete == '':
        return run_episode(env, args)
    return run_fixed_athlete_episode(env, args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discrete", action='store_true',
                        help="use discrete actions")
    parser.add_argument("-fa", "--fixed_athlete", type=str, default='',
                        help=(
                            "use fixed athlete policy from: "
                            f"{list(policy.ATHLETE_POLICIES)}. Or use '' for "
                            "full multi-agent env (default='')"
                        ))
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Verbosity level (default=1)")
    parser.add_argument("-r", "--render", action="store_true",
                        help="Render environment")
    main(parser.parse_args())
