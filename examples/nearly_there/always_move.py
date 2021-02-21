"""PPO Agent for the NearlyThere Env with coach policy fixed """
import numpy as np

from bdgym.envs import NearlyThereEnv
from bdgym.wrappers import FixedPolicyMAWrapper
from bdgym.agents.base_policy import BasePolicy


class TrueCoachPolicy(BasePolicy):
    """Fixed Coach Policy for the NearlyThere Env

    Will supply the coach's observation of dist_remaining to athlete.
    """

    def __init__(self, offset: float = -0.1):
        super().__init__()
        assert -1.0 <= offset <= 1.0
        self.offset = offset

    def get_action(self, obs):
        """Inherited """
        return max(0.0, min(1.0, obs[1] + self.offset))


def eval_model(env, steps=100000):
    """Evaluate model """
    env.reset()
    ep_num = 0
    ep_returns = []
    ep_return = 0
    for _ in range(steps):
        action = 0
        _, reward, done, _ = env.step(action)
        ep_return += reward
        if done:
            ep_returns.append(ep_return)
            env.reset()
            ep_num += 1
            ep_return = 0

    print(f"\n{'-'*60}Training Complete{'-'*60}")
    print(
        f"Mean return = {np.mean(ep_returns):.4f} +/- "
        f"{np.std(ep_returns):.4f}"
    )
    env.close()


if __name__ == "__main__":
    ne_env = NearlyThereEnv()
    ne_env = FixedPolicyMAWrapper(ne_env, {ne_env.COACH: TrueCoachPolicy()})
    eval_model(ne_env)
