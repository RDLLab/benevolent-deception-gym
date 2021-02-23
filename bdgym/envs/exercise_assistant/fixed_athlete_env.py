"""The Exercise Assistant Environment with Fixed Athlete Policy """
from typing import Tuple, Dict

import numpy as np

from bdgym.envs.exercise_assistant.policy import AthletePolicy
from bdgym.envs.exercise_assistant.env import ExerciseAssistantEnv
from bdgym.envs.exercise_assistant.discrete_env import \
    DiscreteExerciseAssistantEnv


class FixedAthleteExerciseAssistantEnv(ExerciseAssistantEnv):
    """ The Exercise Assistant Environment with Fixed Athlete Policy """

    def __init__(self, athlete_policy: AthletePolicy):
        super().__init__()
        self.athlete_policy = athlete_policy
        self.action_space = self.action_space[self.ASSISTANT_IDX]
        self.observation_space = self.observation_space[self.ASSISTANT_IDX]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        athlete_obs, _, _, _ = super().step(action)
        athlete_action = self.athlete_policy.get_action(athlete_obs)
        return super().step(athlete_action)

    def discrete_assistant(self) -> bool:
        return False


class DiscreteFixedAthleteExerciseAssistantEnv(DiscreteExerciseAssistantEnv):
    """ The Discrete Exercise Assistant Env with Fixed Athlete Policy """

    def __init__(self, athlete_policy: AthletePolicy):
        super().__init__()
        self.athlete_policy = athlete_policy
        self.action_space = self.action_space[self.ASSISTANT_IDX]
        self.observation_space = self.observation_space[self.ASSISTANT_IDX]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        athlete_obs, _, _, _ = super().step(action)
        athlete_action = self.athlete_policy.get_action(athlete_obs)
        return super().step(athlete_action)

    def discrete_assistant(self) -> bool:
        return True
