"""A keyboard agent for the Percieved Effort Environment """
from bdgym.envs import NearlyThereEnv
from bdgym.envs.nearly_there import AthleteAction

import numpy as np


LINE_BREAK = "-" * 60


def display_init_obs(obs_n):
    """Display initial observation """
    print(LINE_BREAK)
    print("ATHLETE")
    print(
        f"\tObservation: energy={obs_n[0][0]:.4f} "
        f"coach_dist={obs_n[0][1]:.4f}"
    )
    print()
    print("COACH")
    print(f"\tObservation: energy={obs_n[1][0]:.4f} dist={obs_n[1][1]:.4f}")


def display_step(steps, action_n, obs_n, rew_n, done_n):
    """Display a single step """
    print(LINE_BREAK)
    print(f"Step = {steps}")
    print(LINE_BREAK)
    print("ATHLETE")
    print(f"\tAction: {str(AthleteAction(action_n[0]))}")
    print(
        f"\tObservation: energy={obs_n[0][0]:.4f} "
        f"coach_dist={obs_n[0][1]:.4f}"
    )
    print(f"\tReward: {rew_n[0]}")
    print(f"\tDone: {done_n[0]}")
    print()
    print("COACH")
    print(f"\tAction: dist_comm={action_n[1]}")
    print(f"\tObservation: energy={obs_n[1][0]:.4f} dist={obs_n[1][1]:.4f}")
    print(f"\tReward: {rew_n[1]}")
    print(f"\tDone: {done_n[1]}")


def display_episode_end(steps, total_return, final_state):
    """Display end of episode message """
    print(LINE_BREAK)
    print("Episode Complete")
    print(LINE_BREAK)
    print(f"Steps = {steps}")
    print(f"Energy remaining = {final_state[0]:.4f}")
    print(f"Distance to goal = {final_state[1]:.4f}")
    print(f"Athlete return = {total_return[0]}")
    print(f"Coach return = {total_return[1]}")
    print(LINE_BREAK)


def get_athlete_action():
    """Get action from user """
    print(LINE_BREAK)
    print("Select Athlete Action from:")

    output = ["Actions:"]
    for i in range(len(AthleteAction)):
        output.append(f"{i}={str(AthleteAction(i))}")
    print(" ".join(output))

    while True:
        try:
            idx = int(input("Choose action number: "))
            action = AthleteAction(idx)
            print(f"Performing: {str(action)}")
            return action
        except ValueError:
            print("Invalid choice. Try again.")


def get_coach_action():
    """Get action from user """
    print(LINE_BREAK)
    print("Select Coach Action (distance to report in [0.0, 1.0]:")
    while True:
        try:
            dist = float(input("Distance: "))
            assert 0.0 <= dist <= 1.0
            print(f"Performing: report_dist={dist}")
            return dist
        except (ValueError, AssertionError):
            print("Invalid choice. Try again.")


def run_episode(env):
    """Run a single episode """
    obs_n = env.reset()
    display_init_obs(obs_n)

    total_return = np.zeros(2)
    steps = 0
    done_n = [False, False]
    while not all(done_n):
        athlete_action = get_athlete_action()
        coach_action = get_coach_action()
        action_n = [athlete_action, coach_action]
        obs_n, rew_n, done_n, _ = env.step(action_n)

        display_step(steps, action_n, obs_n, rew_n, done_n)
        total_return += rew_n
        steps += 1

    display_episode_end(steps, total_return, env.state)
    return steps, total_return


if __name__ == "__main__":

    ne_env = NearlyThereEnv()
    run_episode(ne_env)
