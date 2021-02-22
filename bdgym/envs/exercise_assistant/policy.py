"""Fixed Athlete Policies """
from abc import ABC, abstractmethod

import numpy as np

from bdgym.envs.exercise_assistant.action import AthleteAction


class AthletePolicy(ABC):
    """Fixed Athlete Policy for Exercise Assistant Environment """

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> AthleteAction:
        """Get athlete action for given observation

        Parameters
        ----------
        obs : np.ndarray
            the athlete observation

        Returns
        -------
        AthleteAction
           the athletes action
        """


class RandomAthletePolicy(AthletePolicy):
    """Athlete Policy that selects action uniformly at random """

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        return np.random.choice(AthleteAction)


class GreedyAthletePolicy(AthletePolicy):
    """Athlete Policy that performs rep if percieved energy >= threshold """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        if obs[0] >= self.threshold:
            return AthleteAction.PERFORM_REP
        return AthleteAction.END_SET


class ObedientAthletePolicy(AthletePolicy):
    """Athlete Policy that selects follows the assistant recommendation.

    Specifically, if the assistant recommendation is >= 0.5 then athlete will
    END_SET, otherwise athlete will PERFORM_REP.
    """

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        if obs[3] >= 0.5:
            return AthleteAction.END_SET
        return AthleteAction.PERFORM_REP


ATHLETE_POLICIES = {
    'greedy': GreedyAthletePolicy,
    'random': RandomAthletePolicy,
    'obedient': ObedientAthletePolicy
}
