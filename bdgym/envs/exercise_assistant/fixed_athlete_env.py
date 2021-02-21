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

    def reset(self) -> np.ndarray:
        assistant_obs, _ = super().reset()
        return assistant_obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        athlete_obs, _, _, _ = super().step(action)

        athlete_action = self.athlete_policy.get_action(athlete_obs)
        return super().step(athlete_action)


class DiscreteFixedAthleteExerciseAssistantEnv(DiscreteExerciseAssistantEnv):
    """ The Discrete Exercise Assistant Env with Fixed Athlete Policy """

    def __init__(self, athlete_policy: AthletePolicy):
        super().__init__()
        self.athlete_policy = athlete_policy

    def reset(self) -> np.ndarray:
        assistant_obs, _ = super().reset()
        return assistant_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        athlete_obs, _, _, _ = super().step(action)

        athlete_action = self.athlete_policy.get_action(athlete_obs)
        return super().step(athlete_action)
