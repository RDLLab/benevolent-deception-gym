"""Percieved Effort Environment Module """
from enum import IntEnum

import gym
import numpy as np
from gym import spaces


class CoachSignal(IntEnum):
    """Coach signals """
    NOSIGNAL = 0
    GREEN = 1
    RED = 2

    def __str__(self):
        return f"{self.name}"


class AthleteAction(IntEnum):
    """Athlete action indices """
    MOVE = 0
    STOP = 1

    def __str__(self):
        return f"{self.name}"

    def __repre__(self):
        return f"<{self.__class__.name}> {self.name}"


class PercievedEffortEnv(gym.Env):
    """The Percieved Effort Environment """

    metadata = {
        'render.modes': ['human'],
    }

    athlete = 0
    coach = 1

    def __init__(self):
        self.num_agents = 2

        self.action_space = {
            self.athlete: spaces.Discrete(2),
            self.coach: spaces.Discrete(3)
        }

        self.observation_space = {
            self.athlete: spaces.Tuple((
                spaces.Box(low=0.0, high=1.0, shape=()),
                spaces.Discrete(3)
            )),
            self.coach: spaces.Box(low=0.0, high=1.0, shape=())
        }

        self.state = 1.0
        self.energy_cost_dist = lambda: np.random.normal(0.05, 0.025)
        self.athlete_obs_noise = lambda: np.random.normal(-0.1, 0.03)
        self.coach_obs_noise = lambda: np.random.normal(0.0, 0.02)
        self.overexertion_penalty = -100.0
        self.move_reward = 1.0

    def reset(self):
        """Reset the environment

        Returns
        -------
        List
            initial observation for each agent
        """
        self.state = 1.0
        obs = [
            (
                min(1.0, self.state + self.athlete_obs_noise()),
                CoachSignal.NOSIGNAL
            ),
            min(1.0, self.state + self.coach_obs_noise())
        ]
        return obs

    def step(self, action_n):
        """Perform a single step in environment

        Parameters
        ----------
        action_n : List[Action]
            action for each agent

        Returns
        -------
        List[]
            observation for each agent
        List[float]
            reward for each agent
        List[bool]
            terminal signal for each agent
        List[dict]
            auxiliary information for each agent
        """
        assert len(action_n) == self.num_agents

        if action_n[self.athlete] == AthleteAction.MOVE:
            self.state -= max(0.0001, self.energy_cost_dist())

        obs_n = self._get_obs(action_n)
        rew_n = self._get_reward(action_n)
        done_n = self._get_done(action_n)
        info_n = [{}, {}]

        return obs_n, rew_n, done_n, info_n

    def _get_obs(self, action_n):
        """Get observation for each agent after step"""
        obs = [
            (
                max(0.0, min(1.0, self.state + self.athlete_obs_noise())),
                CoachSignal(action_n[self.coach])
            ),
            max(0.0, min(1.0, self.state + self.coach_obs_noise()))
        ]
        return obs

    def _get_reward(self, action_n):
        """Get reward for each agent after step """
        if self.state <= 0.0:
            reward = self.overexertion_penalty
        elif action_n[self.athlete] == AthleteAction.MOVE:
            reward = self.move_reward
        else:
            reward = 0.0
        return [reward] * self.num_agents

    def _get_done(self, action_n):
        """Get whether episode is finished or not """
        if action_n[self.athlete] == AthleteAction.STOP:
            done = True
        else:
            done = self.state <= 0.0
        return [done] * self.num_agents

    def render(self, mode='human'):
        """Render the environment

        Parameters
        ----------
        mode : str
            render mode to use
        """
        assert mode in self.metadata['render.modes']
        print(f"State={self.state:.4f}")
