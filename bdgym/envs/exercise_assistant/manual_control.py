"""Functions and classes for manual control of environment """
from typing import TYPE_CHECKING, Optional

import pygame as pg

from bdgym.envs.exercise_assistant.action import DiscreteAssistantAction

if TYPE_CHECKING:
    from bdgym.envs.exercise_assistant.env import ExerciseAssistantEnv


class GameQuitException(Exception):
    """Exception for when game has been quit """
    pass


class AssistantEventHandler:
    """Event Handler for assistant actions """

    KEY_DISCRETE_ACTION_MAP = {
        pg.K_q: DiscreteAssistantAction.DEC_PERFORM_REP,
        pg.K_w: DiscreteAssistantAction.NO_CHANGE_PERFORM_REP,
        pg.K_e: DiscreteAssistantAction.INC_PERFORM_REP,
        pg.K_a: DiscreteAssistantAction.DEC_END_SET,
        pg.K_s: DiscreteAssistantAction.NO_CHANGE_END_SET,
        pg.K_d: DiscreteAssistantAction.INC_END_SET
    }
    """Map from keyboard key to discrete assistant action """

    @classmethod
    def handle_discrete_events(cls,
                               env: 'ExerciseAssistantEnv'
                               ) -> Optional[DiscreteAssistantAction]:
        """Handle events for discrete Assistant environment """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                env.close()
                raise GameQuitException()
            action = cls.handle_discrete_action_event(event)
            if action is not None:
                return action
        return None

    @classmethod
    def handle_discrete_action_event(cls,
                                     event: pg.event.EventType
                                     ) -> Optional[DiscreteAssistantAction]:
        """Handle Pygame event for discrete assistant action env """
        if (
            event.type != pg.KEYDOWN
            or event.key not in cls.KEY_DISCRETE_ACTION_MAP
        ):
            return None

        return cls.KEY_DISCRETE_ACTION_MAP[event.key]
