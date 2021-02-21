"""Nearly there environment """
from enum import IntEnum

import gym
import numpy as np
from gym import spaces


class AthleteAction(IntEnum):
    """Athlete action indices """
    MOVE = 0
    STOP = 1

    def __str__(self):
        return f"{self.name}"


class NearlyThereEnv(gym.Env):
    """The Nearly There Environment """

    metadata = {
        'render.modes': ['human'],
    }

    ATHLETE = 0
    COACH = 1
    ENERGY_STATE = 0
    DIST_STATE = 1

    MIN_ENERGY = 0.0
    MAX_ENERGY = 1.0
    GOAL_DIST = 0.0
    START_DIST = 1.0
    MIN_ENERGY_COST = 0.0001
    MIN_PROGRESS = 0.0001
    OVEREXERTION_PENALTY = -1
    GOAL_REWARD = 1

    def __init__(self):
        self.num_agents = 2

        self.action_space = {
            self.ATHLETE: spaces.Discrete(len(AthleteAction)),
            self.COACH: spaces.Box(
                low=self.GOAL_DIST, high=self.START_DIST, shape=(1,)
            )
        }

        self.observation_space = {
            self.ATHLETE: spaces.Box(
                low=np.array([self.MIN_ENERGY, self.GOAL_DIST]),
                high=np.array([self.MAX_ENERGY, self.START_DIST])
            ),
            self.COACH: spaces.Box(
                low=np.array([self.MIN_ENERGY, self.GOAL_DIST]),
                high=np.array([self.MAX_ENERGY, self.START_DIST])
            )
        }

        self.state = [self.MAX_ENERGY, self.START_DIST]
        self.energy_cost_dist = lambda: np.random.normal(0.09, 0.05)
        self.progress_dist = lambda: np.random.normal(0.1, 0.025)
        self.athlete_obs_noise = lambda: np.random.normal(-0.1, 0.03)
        self.coach_obs_noise = lambda: np.random.normal(0.0, 0.02)

    def reset(self):
        """Reset the environment

        Returns
        -------
        List
            initial observation for each agent
        """
        self.state = [self.MAX_ENERGY, self.START_DIST]
        athlete_obs = np.array(
            [
                min(
                    self.MAX_ENERGY,
                    self.state[self.ENERGY_STATE] + self.athlete_obs_noise()
                ),
                self.START_DIST
            ],
            dtype=np.float32
        )

        coach_obs = np.array(
            [
                min(
                    self.MAX_ENERGY,
                    self.state[self.ENERGY_STATE] + self.coach_obs_noise()
                ),
                self.START_DIST
            ],
            dtype=np.float32
        )

        obs_n = [athlete_obs, coach_obs]
        return obs_n

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
        assert self.GOAL_DIST <= action_n[self.COACH] <= self.START_DIST

        if action_n[self.ATHLETE] == AthleteAction.MOVE:
            energy_cost = max(self.MIN_ENERGY_COST, self.energy_cost_dist())
            self.state[self.ENERGY_STATE] -= energy_cost
            progress = max(self.MIN_PROGRESS, self.progress_dist())
            self.state[self.DIST_STATE] -= progress

        obs_n = self._get_obs(action_n)
        rew_n = self._get_reward()
        done_n = self._get_done(action_n)
        info_n = self._get_info()

        return obs_n, rew_n, done_n, info_n

    def _get_obs(self, action_n):
        """Get observation for each agent after step"""
        energy_rem = self.state[self.ENERGY_STATE]
        coach_signal = action_n[self.COACH]

        if isinstance(coach_signal, np.ndarray):
            coach_signal = coach_signal.tolist()[0]
        elif isinstance(coach_signal, (list, tuple)):
            coach_signal = coach_signal[0]

        athlete_obs = np.array(
            [
                max(
                    self.MIN_ENERGY,
                    min(self.MAX_ENERGY, energy_rem + self.athlete_obs_noise())
                ),
                coach_signal
            ],
            dtype=np.float32
        )

        coach_obs = np.array(
            [
                max(
                    self.MIN_ENERGY,
                    min(self.MAX_ENERGY, energy_rem + self.coach_obs_noise())
                ),
                self.state[self.DIST_STATE]
            ],
            dtype=np.float32
        )

        obs_n = [athlete_obs, coach_obs]
        return obs_n

    def _get_reward(self):
        """Get reward for each agent after step """
        if self.state[self.ENERGY_STATE] <= self.MIN_ENERGY:
            reward = self.OVEREXERTION_PENALTY
        elif self.state[self.DIST_STATE] <= self.GOAL_DIST:
            reward = self.GOAL_REWARD
        else:
            reward = 0.0
        return [reward] * self.num_agents

    def _get_done(self, action_n):
        """Get whether episode is finished or not """
        done = (
            action_n[self.ATHLETE] == AthleteAction.STOP
            or self.state[self.DIST_STATE] <= self.GOAL_DIST
            or self.state[self.ENERGY_STATE] <= self.MIN_ENERGY
        )
        return [done] * self.num_agents

    def _get_info(self):
        """Get aux information for current step """
        state_info = {
            "energy remaining": self.state[self.ENERGY_STATE],
            "distance to goal": self.state[self.DIST_STATE]
        }
        return [state_info] * self.num_agents

    def render(self, mode='human'):
        """Render the environment

        Parameters
        ----------
        mode : str
            render mode to use
        """
        assert mode in self.metadata['render.modes']
        print(
            f"State: energy={self.state[self.ENERGY_STATE]:.4f} "
            f"distance={self.state[self.DIST_STATE]:.4f}"
        )
