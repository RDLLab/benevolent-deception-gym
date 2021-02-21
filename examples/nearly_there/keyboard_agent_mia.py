"""A keyboard agent for the Nearly There Environment using ther
Multi-Independent-Agent Interface Wrappers
"""
from multiprocessing import Process, Lock

from bdgym.wrappers import MIAWrapper
from bdgym.envs import NearlyThereEnv
from bdgym.envs.nearly_there import AthleteAction


LINE_BREAK = "-" * 60


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


def run_athlete_episode(env, lock):
    """Run a single episode """

    obs = env.reset()
    with lock:
        print(LINE_BREAK)
        print(f"ATHLETE Init Obs: energy={obs[0]:.4f} coach_dist={obs[1]:.4f}")
        print(LINE_BREAK)

    final_return = 0
    steps = 0
    done = False
    while not done:
        with lock:
            # action = get_athlete_action()
            action = 0 if obs[0] > 0.0 else 1
        obs, rew, done, _ = env.step(action)

        step_output = [
            LINE_BREAK,
            f"ATHLETE Step = {steps}",
            f"\tAction: {str(AthleteAction(action))}",
            f"\tObservation: energy={obs[0]:.4f} coach_dist={obs[1]:.4f}",
            f"\tReward: {rew}",
            f"\tDone: {done}",
            LINE_BREAK,
        ]
        with lock:
            print("\n".join(step_output))

        final_return += rew
        steps += 1

    end_output = [
        LINE_BREAK,
        "ATHLETE Episode Complete",
        f"Steps = {steps}",
        f"Athlete return = {final_return}",
        LINE_BREAK
    ]
    with lock:
        print("\n".join(end_output))

    # ensure child connections terminated correctly
    env.close()

    return steps, final_return


def run_coach_episode(env, lock: Lock):
    """Run a single episode """

    obs = env.reset()
    with lock:
        print(LINE_BREAK)
        print(f"Coach Init Obs: energy={obs[0]:.4f} dist={obs[1]:.4f}")
        print(LINE_BREAK)

    final_return = 0
    steps = 0
    done = False
    while not done:
        with lock:
            # action = get_coach_action()
            action = obs[1] - 0.01 if obs[0] > 0.0 else obs[1] + 0.05
        obs, rew, done, _ = env.step(action)

        step_output = [
            LINE_BREAK,
            f"COACH Step = {steps}",
            f"\tAction: dist_comm={action:.4f}",
            f"\tObservation: energy={obs[0]:.4f} dist={obs[1]:.4f}",
            f"\tReward: {rew}",
            f"\tDone: {done}",
            LINE_BREAK,
        ]
        with lock:
            print("\n".join(step_output))

        final_return += rew
        steps += 1

    end_output = [
        LINE_BREAK,
        "COACH Episode Complete",
        f"Steps = {steps}",
        f"COACH return = {final_return}",
        LINE_BREAK
    ]
    with lock:
        print("\n".join(end_output))

    # ensures child connections terminated correctly
    env.close()

    return steps, final_return


if __name__ == "__main__":

    ne_env = NearlyThereEnv()
    ne_env = MIAWrapper(ne_env, ne_env.num_agents)
    athlete_ne_env = ne_env.get_agent_env(NearlyThereEnv.ATHLETE)
    coach_ne_env = ne_env.get_agent_env(NearlyThereEnv.COACH)

    mp_lock = Lock()
    athlete_process = Process(
        target=run_athlete_episode, args=(athlete_ne_env, mp_lock)
    )
    coach_process = Process(
        target=run_coach_episode, args=(coach_ne_env, mp_lock)

    )
    athlete_process.start()
    coach_process.start()

    athlete_process.join()
    coach_process.join()

    print(LINE_BREAK)
    print("EPISODE COMPLETE")
    print(f"Energy remaining = {ne_env.env.state[0]:.4f}")
    print(f"Distance to goal = {ne_env.env.state[1]:.4f}")
    print(LINE_BREAK)
    ne_env.close()
