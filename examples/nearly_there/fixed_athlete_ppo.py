"""PPO Agent for the NearlyThere Env with coach policy fixed """
import numpy as np
from stable_baselines3 import PPO

from bdgym.envs import NearlyThereEnv
from bdgym.envs.nearly_there import AthleteAction
from bdgym.wrappers import FixedPolicyMAWrapper
from bdgym.agents.base_policy import BasePolicy


class AthletePolicy(BasePolicy):
    """Fixed Athlete Policy for the NearlyThere Env

    Will STOP when:

        observed energy remaining < x and coach signal for distance
        remaining > y

    otherwise will MOVE
    """

    def __init__(self, energy_limit: float = 0.1, dist_cutoff: float = 0.05):
        super().__init__()
        assert 0.0 <= energy_limit <= 1.0
        assert 0.0 <= dist_cutoff <= 1.0
        self.energy_limit = energy_limit
        self.dist_cutoff = dist_cutoff

    def get_action(self, obs):
        """Inherited """
        if obs[0] > self.energy_limit or obs[1] < self.dist_cutoff:
            return AthleteAction.MOVE
        return AthleteAction.STOP


class AggressiveAthletePolicy(BasePolicy):
    """Fixed Athlete Policy for the NearlyThere Env

    Will only STOP when:

       energy remaining - coach signal for distance remaining < y

    otherwise will MOVE
    """

    def __init__(self,
                 energy_limit: float = 0.01,
                 energy_dist_diff: float = -0.1):
        super().__init__()
        assert 0.0 <= energy_limit <= 1.0
        assert -1.0 <= energy_dist_diff <= 1.0
        self.energy_limit = energy_limit
        self.energy_dist_diff = energy_dist_diff

    def get_action(self, obs):
        """Inherited """
        if obs[0] - obs[1] < self.energy_dist_diff:
            return AthleteAction.STOP
        return AthleteAction.MOVE


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

    # athlete_policy = AthletePolicy
    athlete_policy = AggressiveAthletePolicy

    ne_env = NearlyThereEnv()
    ne_env = FixedPolicyMAWrapper(ne_env, {ne_env.ATHLETE: athlete_policy()})

    ne_eval_env = NearlyThereEnv()
    ne_eval_env = FixedPolicyMAWrapper(
        ne_eval_env, {ne_env.ATHLETE: athlete_policy()}
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
        tb_log_name=f"PPO_{athlete_policy.__name__}"
    )

    eval_model(ppo_model, ne_env)
