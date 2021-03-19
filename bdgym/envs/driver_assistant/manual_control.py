"""Functions and class for manual control of environment """
from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame as pg

from highway_env.envs.common.graphics import EventHandler
from highway_env.envs.common.action import ActionType, Action

from bdgym.envs.driver_assistant.action import AssistantDiscreteAction

if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class GameQuitException(Exception):
    """Exception for when game has been quit """
    pass


class AssistantEventHandler(EventHandler):
    """Event handler that includes assistant actions """

    KEY_ACTION_MAP = {
        pg.K_RIGHT: ("acceleration", True),
        pg.K_LEFT: ("acceleration", False),
        pg.K_UP: ("steering", False),
        pg.K_DOWN: ("steering", True),
        pg.K_q: ("x", True),
        pg.K_a: ("x", False),
        pg.K_w: ("y", True),
        pg.K_s: ("y", False),
        pg.K_e: ("vx", True),
        pg.K_d: ("vx", False),
        pg.K_r: ("vy", True),
        pg.K_f: ("vy", False),
    }
    """Map from keyboard key to (action key, increase) tuple.
    Where increase is a bool that specifies if keyboard key increases (True) or
    decreases (False) the given action offset
    """

    CONTROL_ACTIONS = [pg.K_RIGHT, pg.K_LEFT, pg.K_UP, pg.K_DOWN]
    """Action Keys corresponding to 'acceleration' and 'steering' action """

    @classmethod
    def get_discrete_action(cls, env: 'DriverAssistantEnv') -> Action:
        """Get action via manual control.

        This simply checks if any key has been entered and parses it into
        an action. If no key has been pressed then returns the default NOOP
        action.
        """
        assistant_action_type = env.action_type.assistant_action_type
        for event in pg.event.get():
            if event.type == pg.QUIT:
                env.close()
                raise GameQuitException()
            action = cls.handle_get_discrete_action_event(
                assistant_action_type, event
            )
            if action is not None:
                return action
        return cls.get_noop_discrete_action(assistant_action_type)

    @classmethod
    def get_noop_discrete_action(cls,
                                 action_type: AssistantDiscreteAction
                                 ) -> Action:
        """Get the NOOP Discrete Action """
        return np.full(
            action_type.ASSISTANT_DISCRETE_ACTION_SPACE_SIZE,
            action_type.NOOP,
            dtype=np.float32
        )

    @classmethod
    def handle_get_discrete_action_event(cls,
                                         action_type: AssistantDiscreteAction,
                                         event: pg.event.EventType
                                         ) -> Optional[Action]:
        """Handle getting Discrete Assistant Action """
        if (
                event.type not in [pg.KEYDOWN, pg.KEYUP]
                or event.key not in cls.KEY_ACTION_MAP
        ):
            return

        action = cls.get_noop_discrete_action(action_type)

        a_key, increase = cls.KEY_ACTION_MAP[event.key]
        a_idx = action_type.ASSISTANT_DISCRETE_ACTION_INDICES[a_key]
        if event.type == pg.KEYDOWN:
            if increase:
                action[a_idx] = action_type.UP
            else:
                action[a_idx] = action_type.DOWN
        elif event.type == pg.KEYUP and event.key in cls.CONTROL_ACTIONS:
            action[a_idx] = action_type.NOOP
        return action

    @classmethod
    def handle_event(cls,
                     action_type: ActionType,
                     event: pg.event.EventType) -> None:
        """ Overrides parent """
        assistant_action_type = action_type.assistant_action_type
        if isinstance(assistant_action_type, AssistantDiscreteAction):
            cls.handle_assistant_discrete_action_event(
                assistant_action_type, event
            )
        else:
            super().handle_event(assistant_action_type, event)

    @classmethod
    def handle_assistant_discrete_action_event(
            cls,
            action_type: AssistantDiscreteAction,
            event: pg.event.EventType) -> None:
        """Event handler for Assistant Discrete Actions """
        if (
                event.type not in [pg.KEYDOWN, pg.KEYUP]
                or event.key not in cls.KEY_ACTION_MAP
        ):
            return

        action = cls.handle_get_discrete_action_event(action_type, event)
        action_type.act(action)
