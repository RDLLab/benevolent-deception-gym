"""Script for getting average performance of the different athlete policies """

import gym
import numpy as np

import bdgym  # noqa: F401 pylint: disable=unused-import
import bdgym.envs.exercise_assistant.policy as policy


LINE_BREAK = "=" * 60
SMALL_LINE_BREAK = "-" * 30

TEST_ENVS = {
    "ExerciseAssistantOA-v0": 100,
    "ExerciseAssistantIA-v0": 100,
    "ExerciseAssistantHA-v0": 100
}

ASSISTANT_POLICIES = [
    policy.DoNothingDiscreteAssistantPolicy(),
    policy.GreedyDiscreteAssistantPolicy(0.15)
]


def run_episode(env_name: str, assistant_policy):
    """Run episode on env with given assistant policy """
    env = gym.make(env_name)

    obs = env.reset()
    total_return = 0.0
    steps = 0
    done = False
    while not done:
        action = assistant_policy.get_action(obs)
        obs, rew, done, _ = env.step(action)
        total_return += rew
        steps += 1

    return total_return, steps, env.athlete_overexerted()


def run_env(env_name: str, assistant_policy):
    """Run multiple episodes of env """
    num_episodes = TEST_ENVS[env_name]

    ep_returns = []
    ep_steps = []
    ep_overexertions = 0
    for _ in range(num_episodes):
        ep_return, steps, overexerted = run_episode(env_name, assistant_policy)
        ep_returns.append(ep_return)
        ep_steps.append(steps)
        ep_overexertions += int(overexerted)

    print(f"\nTest complete for env={env_name}")
    print(SMALL_LINE_BREAK)
    print(
        f"Mean return = {np.mean(ep_returns):.3f} +/- {np.std(ep_returns):.3f}"
    )
    print(f"Mean steps = {np.mean(ep_steps):.3f} +/- {np.std(ep_steps):.3f}")
    print(f"Overexertion prob = {ep_overexertions / num_episodes:.3f}")


def run_assistant_policy(assistant_policy):
    """Run envs with assistant policy """
    print(LINE_BREAK)
    print(f"Testing assistant policy: {assistant_policy.__class__.__name__}")
    print(LINE_BREAK)
    for env_name in TEST_ENVS:
        run_env(env_name, assistant_policy)


def run():
    """Run evals """
    for assistant_policy in ASSISTANT_POLICIES:
        run_assistant_policy(assistant_policy)


if __name__ == "__main__":
    run()
