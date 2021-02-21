"""An environment wrapper for multi-agent gym environments

It returns only the observations for a single agent
"""
import gym


class SingleAgentWrapper(gym.Wrapper):
    """A single agent wrapper for multi-agetnt gym environments """

    def __init__(self, env: gym.Env, ego_agent_id: int):
        assert 0 <= ego_agent_id < len(env.action_space), \
            "Agent ID for SingleAgentWrapper outside number of agents in env"
        super().__init__(env)
        self.ego_agent_id = ego_agent_id
        self.action_space = self.env.action_space[self.ego_agent_id]
        self.observation_space = self.env.observation_space[self.ego_agent_id]

    def step(self, action):
        """Perform step in environment for ego agent """
        obs, rew, done, info = self.env.step_agent(action, self.ego_agent_id)
        # print(f"Agent {self.ego_agent_id} step:")
        # print(
        #     f"action={action}\nobs={obs}\nrew={rew}\ndone={done}\ninfo={info}"
        # )
        return obs, rew, done, info

    def reset(self, **kwargs):
        """Reset environment and get obs of ego agent """
        obs = self.env.reset_agent(self.ego_agent_id, **kwargs)
        # print(f"Agent {self.ego_agent_id} reset:")
        # print(f"init obs={obs}")
        return obs

    def close(self):
        """Close env """
        self.env.close_conns()
