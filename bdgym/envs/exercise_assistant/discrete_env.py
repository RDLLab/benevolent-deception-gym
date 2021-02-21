"""The Exercise Assistant Environment where Assistant has Discrete Actions. """
from typing import Tuple, Union, Dict

import numpy as np
from gym import spaces

import bdgym.envs.utils as utils
from bdgym.envs.exercise_assistant.env import ExerciseAssistantEnv
from bdgym.envs.exercise_assistant.action import DiscreteAssistantAction


class DiscreteExerciseAssistantEnv(ExerciseAssistantEnv):
    """The Exercise Assistant Environment with Discrete Assistant Actions.

    Note everything about the environment is unchanged from the
    ExerciseAssistantEnv except the Assistant's observation and action spaces.

    Assistant Properties
    --------------------
    Observation:
        Type: Box(3)
        Num   Observation                           Min       Max
        0     Athlete Energy Level                  0.0       1.0
        1     Proportion of sets complete           0.0       1.0
        2     Athlete Action                        0.0       1.0
        3     Energy Signal Offset                  0.0       1.0

        The assistant observations are unchanged except for the additional
        observation of the current energy signal offset value.

        Note, energy signal offset observation is linearly mapped to
        [0.0, 1.0] from the full range [-1.0, 1.0]

    Actions:
        Type: Discrete(6)
        Num   Action
        0     (increase, perform-rep)
        1     (no-change, perform-rep)
        2     (decrease, perform-rep)
        3     (increase, end-set)
        4     (no-change, end-set)
        5     (decrease, end-set)

        The Discrete Assistant Action space is the joint space of the signal
        offset actions ('increase', 'no-change', 'decrease') and the
        recommendation actions ('perform-rep', 'end-set').

        The signal offset actions modify how much the assistant changes the
        energy level signal that is reported to the athlete. Each 'increase'
        action increases the offset by +OFFSET_STEP and each 'decrease'
        action decreases the offset by -OFFSET_STEP. Initially the offset
        is 0.0.

        For example, if the assistant has used the 'increase' twice and no
        'decrease' action then the current offset would be +0.1. Then if the
        assistant observes that the athlete has 0.65 of their energy remaining
        the signal the coach will send to the athlete is that they have
        0.65 + 0.1 = 0.75 energy remaining.
    """

    OFFSET_STEP = 0.05
    """The amount offset is changed per assistant offset change action """

    OFFSET_INIT = 0.0
    """The initial offset, at the start of the episode """

    OFFSET_MIN = -1.0
    """The minimum possible offset value """

    OFFSET_MAX = 1.0
    """The maximum possible offset value """

    def __init__(self):
        super().__init__()
        self.action_space[self.ASSISTANT_IDX] = spaces.Discrete(
            len(DiscreteAssistantAction)
        )
        self.observation_space[self.ASSISTANT_IDX] = spaces.Box(
            low=0.0, high=1.0, shape=(4,)
        )
        self._current_offset = 0.0

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        assistant_obs, athlete_obs = super().reset()
        self._current_offset = 0.0
        assistant_obs = self._convert_assistant_obs(assistant_obs)
        self._last_obs[self.ASSISTANT_IDX] = assistant_obs
        return assistant_obs, athlete_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.next_agent == self.ASSISTANT_IDX:
            self._update_assistant_offset(action)
            action = self._convert_assistant_action(action)
            return super().step(action)

        obs, reward, done, info = super().step(action)
        obs = self._convert_assistant_obs(obs)
        self._last_obs[self.next_agent] = obs
        return obs, reward, done, info

    def _update_assistant_offset(self, action: int):
        if DiscreteAssistantAction.is_increase(action):
            self._current_offset = min(
                self.OFFSET_MAX, self._current_offset + self.OFFSET_STEP
            )
        elif DiscreteAssistantAction.is_decrease(action):
            self._current_offset = max(
                self.OFFSET_MIN, self._current_offset - self.OFFSET_STEP
            )

    def _convert_assistant_action(self, action: int) -> np.ndarray:
        # convert discrete action into continuous (signal, recommendation)
        # action, with ranges [-1.0, 1.0]
        signal = self._last_obs[self.ASSISTANT_IDX][0] + self._current_offset
        signal = np.clip(signal, self.MIN_ENERGY, self.MAX_ENERGY)
        signal = utils.lmap(
            signal, [self.MIN_ENERGY, self.MAX_ENERGY], [-1.0, 1.0]
        )

        if DiscreteAssistantAction.is_perform_rep(action):
            recommendation = -1.0
        else:
            recommendation = 1.0
        return np.array([signal, recommendation], dtype=np.float32)

    def _convert_assistant_obs(self, obs: np.ndarray) -> np.ndarray:
        # ensure offset is mapped to [0.0, 1.0] and concatenate to obs
        offset = utils.lmap(
            self._current_offset, [-1.0, 1.0], [0.0, 1.0]
        )
        return np.concatenate((obs, [offset]))

    @property
    def assistant_offset(self) -> float:
        """The current unnormalized offset of the assistant """
        norm_offset = self.last_assistant_obs[3]
        return utils.lmap(norm_offset, [0.0, 1.0], [-1.0, 1.0])
