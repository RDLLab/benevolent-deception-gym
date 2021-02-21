"""Action classes and function for Exercise Assistant Environment """
from enum import IntEnum
from typing import Union

import numpy as np


class AthleteAction(IntEnum):
    """Athlete action indices """
    PERFORM_REP = 0
    END_SET = 1

    def __str__(self):
        return f"{self.name}"


class DiscreteAssistantAction(IntEnum):
    """Discrete Assistant Action indices """
    INC_PERFORM_REP = 0
    NO_CHANGE_PERFORM_REP = 1
    DEC_PERFORM_REP = 2
    INC_END_SET = 3
    NO_CHANGE_END_SET = 4
    DEC_END_SET = 5

    @classmethod
    def is_perform_rep(cls, action: int) -> bool:
        """Check if given action includes the perform rep action """
        return action in (
            cls.INC_PERFORM_REP,
            cls.NO_CHANGE_PERFORM_REP,
            cls.DEC_PERFORM_REP
        )

    @classmethod
    def is_increase(cls, action: int) -> bool:
        """Check if given action includes the 'increase' action """
        return action in (
            cls.INC_PERFORM_REP,
            cls.INC_END_SET
        )

    @classmethod
    def is_decrease(cls, action: int) -> bool:
        """Check if given action includes the 'decrease' action """
        return action in (
            cls.DEC_PERFORM_REP,
            cls.DEC_END_SET
        )

    def __str__(self):
        return f"{self.name}"


def assistant_action_str(action: Union[int, np.ndarray]) -> str:
    """Get string representation of assistant action

    Parameters
    ----------
    action : Union[int, np.ndarray]
        the assistant action

    Returns
    -------
    str
        a human readable string representation of the action
    """
    if isinstance(action, (int, np.integer)):
        return str(DiscreteAssistantAction(action))
    return f"Energy Signal = {action[0]:.3f}, recommendation = {action[1]:.4f}"
