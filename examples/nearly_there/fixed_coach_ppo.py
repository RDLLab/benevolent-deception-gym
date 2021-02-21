"""PPO Agent for the NearlyThere Env with coach policy fixed """
import numpy as np
from stable_baselines3 import PPO

from bdgym.envs import NearlyThereEnv
from bdgym.wrappers import FixedPolicyMAWrapper
from bdgym.agents.base_policy import BasePolicy


class CoachPolicy(BasePolicy):
    """Fixed Coach Policy for the NearlyThere Env

    Will return:

        max(0.0, min(1.0, dist_remaining - x)) while dist_remaining > 0.25
        or athlete energy - dis_remaining > 0.1

    otherwise will return

        max(0.0, min(1.0, dist_remaining + 2x)).

    Where x is the encouragement parameter
    """

    def __init__(self, encouragment: float = 0.1, offset: float = -0.1):
        super().__init__()
        assert -1.0 <= encouragment <= 1.0
        self.encouragement = encouragment
        self.offset = offset

    def get_action(self, obs):
        """Inherited """
        if obs[1] > 0.25 or obs[0] - obs[1] > 0.1:
            return max(
                0.0, min(1.0, obs[1] - self.encouragement + self.offset)
            )
        return max(0.0, min(1.0, obs[1] + 2*self.encouragement + self.offset))


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


def eval_model(model, env, num_eps=10):
    """Evaluate model """
    ep_returns = []
    for e in range(num_eps):
        print(f"\nEpisode={e}\n{'-'*60}")
        ep_return = 0
        o = env.reset()
        done = False
        i = 0
        while not done:
            action, _ = model.predict(o, deterministic=True)
            o, reward, done, info = env.step(action)
            print(
                f"Step={i}, a={action}, o={o}, r={reward}, done={done}, "
                f"i={info}"
            )
            ep_return += reward
            i += 1

        ep_returns.append(ep_return)
        print(f"Episode={e} End. Return={ep_return}")

    print(f"\n{'-'*60}Training Complete{'-'*60}")
    print(
        f"Mean return = {np.mean(ep_returns):.4f} +/- "
        f"{np.std(ep_returns):.4f}"
    )
    env.close()


if __name__ == "__main__":

    # coach_policy = CoachPolicy
    coach_policy = TrueCoachPolicy

    ne_env = NearlyThereEnv()
    ne_env = FixedPolicyMAWrapper(ne_env, {ne_env.COACH: coach_policy()})

    ne_eval_env = NearlyThereEnv()
    ne_eval_env = FixedPolicyMAWrapper(
        ne_eval_env, {ne_env.COACH: coach_policy()}
    )

    ppo_model = PPO(
        'MlpPolicy',
        ne_env,
        verbose=1,
        tensorboard_log="./results/",
    )
    ppo_model.learn(
        total_timesteps=100000,
        eval_env=ne_eval_env,
        eval_freq=2048,
        n_eval_episodes=100,
        tb_log_name=f"PPO_{coach_policy.__name__}"
    )

    eval_model(ppo_model, ne_env)
