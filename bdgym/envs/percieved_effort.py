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

    ATHLETE = 0
    COACH = 1

    MIN_ENERGY = 0.0
    MAX_ENERGY = 1.0
    MIN_ENERGY_COST = 0.0001
    OVEREXERTION_PENALTY = -100
    MOVE_REWARD = 1

    def __init__(self):
        self.num_agents = 2

        self.action_space = {
            self.ATHLETE: spaces.Discrete(len(AthleteAction)),
            self.COACH: spaces.Discrete(len(CoachSignal))
        }

        self.observation_space = {
            self.ATHLETE: spaces.Tuple((
                spaces.Box(
                    low=self.MIN_ENERGY, high=self.MAX_ENERGY, shape=()
                ),
                spaces.Discrete(len(CoachSignal))
            )),
            self.COACH: spaces.Box(
                low=self.MIN_ENERGY, high=self.MAX_ENERGY, shape=()
            )
        }

        self.state = self.MAX_ENERGY
        self.energy_cost_dist = lambda: np.random.normal(0.05, 0.025)
        self.athlete_obs_noise = lambda: np.random.normal(-0.1, 0.03)
        self.coach_obs_noise = lambda: np.random.normal(0.0, 0.02)

    def reset(self):
        """Reset the environment

        Returns
        -------
        List
            initial observation for each agent
        """
        self.state = self.MAX_ENERGY
        obs = [
            (
                min(self.MAX_ENERGY, self.state + self.athlete_obs_noise()),
                CoachSignal.NOSIGNAL
            ),
            min(self.MAX_ENERGY, self.state + self.coach_obs_noise())
        ]
        return obs

    def step(self, action_n):
        """Perform a single step in environment

        Parameters
        ----------
        action_n : List[int, int]
            action for each agent

        Returns
        -------
        List[(float, int), float]
            observation for each agent
        List[float]
            reward for each agent
        List[bool]
            terminal signal for each agent
        List[dict]
            auxiliary information for each agent
        """
        assert len(action_n) == self.num_agents

        if action_n[self.ATHLETE] == AthleteAction.MOVE:
            self.state -= max(self.MIN_ENERGY_COST, self.energy_cost_dist())

        obs_n = self._get_obs(action_n)
        rew_n = self._get_reward(action_n)
        done_n = self._get_done(action_n)
        info_n = self._get_info()

        return obs_n, rew_n, done_n, info_n

    def _get_obs(self, action_n):
        """Get observation for each agent after step"""
        obs = [
            (
                max(
                    self.MIN_ENERGY,
                    min(self.MAX_ENERGY, self.state + self.athlete_obs_noise())
                ),
                CoachSignal(action_n[self.COACH])
            ),
            max(
                self.MIN_ENERGY,
                min(self.MAX_ENERGY, self.state + self.coach_obs_noise())
            )
        ]
        return obs

    def _get_reward(self, action_n):
        """Get reward for each agent after step """
        if self.state <= self.MIN_ENERGY:
            reward = self.OVEREXERTION_PENALTY
        elif action_n[self.ATHLETE] == AthleteAction.MOVE:
            reward = self.MOVE_REWARD
        else:
            reward = 0.0
        return [reward] * self.num_agents

    def _get_done(self, action_n):
        """Get whether episode is finished or not """
        if action_n[self.ATHLETE] == AthleteAction.STOP:
            done = True
        else:
            done = self.state <= self.MIN_ENERGY
        return [done] * self.num_agents

    def _get_info(self):
        """Get aux information for current step """
        return [{"state": self.state}, {"state": self.state}]

    def render(self, mode='human'):
        """Render the environment

        Parameters
        ----------
        mode : str
            render mode to use
        """
        assert mode in self.metadata['render.modes']
        print(f"State={self.state:.4f}")
