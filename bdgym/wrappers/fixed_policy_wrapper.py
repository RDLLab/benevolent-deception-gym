"""A wrapper for MA Environments where one or more agents use fixed policy """
from typing import Dict

import gym

from bdgym.agents.base_policy import BasePolicy


class FixedPolicyMAWrapper(gym.Wrapper):
    """Wrapper for MA environment where one or more agents use fixed policy

    If all but one agent use a fixed-policy then the wrapped environment
    is a single-agent environment, in terms of the interaction of the
    remaining agent.
    """

    def __init__(self, env: gym.Env, fixed_policies: Dict[int, BasePolicy]):
        super().__init__(env)
        self.num_agents = len(env.action_space)
        self.num_fixed_agents = len(fixed_policies)
        self.num_policy_agents = self.num_agents - self.num_fixed_agents
        self.fixed_policies = fixed_policies
        self.policy_agent_ids = [
            i for i in range(self.num_agents) if i not in self.fixed_policies
        ]

        if self.num_policy_agents == 1:
            agent_id = self.policy_agent_ids[0]
            self.action_space = self.action_space[agent_id]
            self.observation_space = self.observation_space[agent_id]

        self._last_obs_n = []

    def reset(self):
        """Reset the environment """
        self._last_obs_n = self.env.reset()
        pol_obs = [self._last_obs_n[i] for i in self.policy_agent_ids]
        if self.num_policy_agents == 1:
            return pol_obs[0]
        return pol_obs

    def step(self, action):
        """Perform a step in the env """
        if self.num_policy_agents == 1:
            action = [action]
        assert len(action) == self.num_policy_agents

        action_n = [None] * self.num_agents
        for i, a in zip(self.policy_agent_ids, action):
            action_n[i] = a

        for i, pi in self.fixed_policies.items():
            action_n[i] = pi.get_action(self._last_obs_n[i])

        obs_n, rew_n, done_n, info_n = self.env.step(action_n)

        pol_obs, pol_rews, pol_dones, pol_infos = [], [], [], []
        for i in self.policy_agent_ids:
            pol_obs.append(obs_n[i])
            pol_rews.append(rew_n[i])
            pol_dones.append(done_n[i])
            pol_infos.append(info_n[i])

        self._last_obs_n = obs_n

        if self.num_policy_agents == 1:
            return pol_obs[0], pol_rews[0], pol_dones[0], pol_infos[0]

        return pol_obs, pol_rews, pol_dones, pol_infos
