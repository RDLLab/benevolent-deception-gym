"""Classes and functions for the human interface for the Driver Assistant  env.

The human interface is designed to allow a human user manually control
the assistant system. This includes modifying the signal sent to the
driver (i.e. 'x', 'y', 'vx', 'vy') as well as the recommended
acceleratio and steering.

Specifically, at each step the human controlled assistant will send
signals and recommendations that equal to the observed values
(in the case of the signals) or 0.0 (in case of the recommendations)
plus the offset specified by the human user.

The offset for each parameter: 'x', 'y', 'vx', 'vy', 'acceleration', and
'acceleration', is controlled via a slider which can be increased or
deacresed using the following keys:

  parameter    increase   decrease
-----------------------------------
     'x'           Q         A
     'y'           W         S
     'v            E         D
     'vy'          R         F
'acceleration'   Right      Left
  'steering'       Up       Down

Where Right, Left, Up, Down correspond to the arrow keys on the
keyboard.
"""
from typing import TYPE_CHECKING, Tuple, List

import numpy as np
import pygame as pg

from bdgym.envs.driver_assistant.action import \
    AssistantContinuousAction, AssistantContinuousOffsetAction

from highway_env.road.graphics import WorldSurface
from highway_env.envs.common.action import ActionType
from highway_env.envs.common.graphics import EnvViewer, EventHandler

if TYPE_CHECKING:
    from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class DriverAssistantEnvViewer(EnvViewer):
    """A viewer to render a Driver Assistant environment """

    def handle_events(self) -> None:
        """Overrides parent."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                AssistantEventHandler.handle_event(self.env.action_type, event)


class AssistantActionDisplayer:
    """A callable class for displaying the Assistant's actions in viewer """

    IGNORE_FEATURES = ['presence']

    def __init__(self, env: 'DriverAssistantEnv'):
        self.env = env
        self.font = pg.font.Font(None, 26)
        self.surface = pg.Surface(
            (self.env.config["screen_width"], self.env.config["screen_height"])
        )
        self.width = self.surface.get_width()
        self.height = self.surface.get_height()

        # Display for what the Assistant Observes
        self.assistant_obs_parameters = []
        self.obs_ignore_idxs = []
        for i, f in enumerate(self.env.config['observation']['features']):
            if f not in self.IGNORE_FEATURES:
                self.assistant_obs_parameters.append(f)
            else:
                self.obs_ignore_idxs.append(i)

        self.assistant_display = DashboardDisplay(
            "Assistant Observation",
            self.surface,
            width=self.width,
            height=self.height // 3,
            position=(0, 0),
            parameters=self.assistant_obs_parameters
        )

        # Display for the current Assistant action
        self.assistant_action = DashboardDisplay(
            "Assistant Action (Driver observation & action recommendation)",
            self.surface,
            width=self.width,
            height=self.height // 3,
            position=(0, self.height // 3),
            parameters=list(AssistantContinuousAction.ASSISTANT_ACTION_INDICES)
        )

        # Display for what the Driver Observes
        self.driver_display = DashboardDisplay(
            "Driver observation",
            self.surface,
            width=self.width,
            height=self.height // 3,
            position=(0, 2*(self.height // 3)),
            parameters=list(AssistantContinuousAction.ASSISTANT_ACTION_INDICES)
        )

    def _get_assistant_ego_obs(self) -> np.ndarray:
        raw_obs = self.env.last_assistant_obs[0]
        obs = []
        for i in range(raw_obs.shape[0]):
            if i not in self.obs_ignore_idxs:
                obs.append(raw_obs[i])
        return np.array(obs)

    def _get_assistant_action(self) -> np.ndarray:
        return self.env.last_assistant_action

    def _get_driver_obs(self,
                        assistant_obs: np.ndarray,
                        assistant_action: np.ndarray) -> np.ndarray:
        assistant_action_type = self.env.action_type.assistant_action_type
        if isinstance(assistant_action_type, AssistantContinuousAction):
            return assistant_action
        if isinstance(
                assistant_action_type, AssistantContinuousOffsetAction
        ):
            num_obs_features = len(self.assistant_obs_parameters)
            assistant_signal = (
                assistant_obs + assistant_action[:num_obs_features]
            )
            return np.concatenate([assistant_signal, assistant_action[4:]])
        raise ValueError(
            "Unsupported Assistant Action Type for Assistant action display: "
            f"{assistant_action_type}. Either use supported action type or "
            "disable action display with the 'action_display' environment "
            "configuration parameter"
        )

    def __call__(self,
                 agent_surface: pg.Surface,
                 sim_surface: WorldSurface) -> None:
        """Draws the assistants last action on agent_surface """
        assistant_obs = self._get_assistant_ego_obs()
        assistant_action = self._get_assistant_action()
        driver_obs = self._get_driver_obs(assistant_obs, assistant_action)

        self.assistant_display.display(assistant_obs)
        self.assistant_action.display(assistant_action)
        self.driver_display.display(driver_obs)

        agent_surface.blit(self.surface, (0, 0))

    def render_text(self,
                    surface: pg.Surface,
                    pos: Tuple[float, float],
                    text: str,
                    color: Tuple[int, int, int],
                    bgcolor: Tuple[int, int, int]) -> Tuple[float, float]:
        """Render text on surface """
        text_img = self.font.render(text, 1, color, bgcolor)
        surface.blit(text_img, pos)
        return pos[0] + text_img.get_width() + 5, pos[1]


class DashboardDisplay:
    """A surface for displaying parameters and values """

    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN
    BG_COLOR = PURPLE

    def __init__(self,
                 title: str,
                 parent_surface: pg.SurfaceType,
                 width: int,
                 height: int,
                 position: Tuple[int, int],
                 parameters: List[str],
                 font_size: int = 26,
                 text_color: Tuple[int, int, int] = None,
                 bg_color: Tuple[int, int, int] = None):
        self.title = title
        self.parent_surface = parent_surface
        self.width = width
        self.height = height
        self.position = position
        self.parameters = parameters
        self.font = pg.font.Font(None, font_size)
        self.text_color = self.YELLOW if text_color is None else text_color
        self.bg_color = self.BLACK if bg_color is None else bg_color

        # The Dashboard Display surface
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)

        # Positions of each parameter
        self.title_pos = (0.05*self.width, 0.05*self.height)
        self.p_name_y = 0.3 * self.height
        self.p_value_y = 0.65 * self.height
        self.p_x = [0.05*self.width]
        for i in range(1, len(self.parameters)):
            self.p_x.append(self.p_x[i-1] + 0.15*self.width)

    def display(self, values: np.ndarray) -> None:
        """Update the dashboard display """
        self.surface.fill(self.bg_color)

        title_img = self._text_img(self.title, self.GREEN)
        self.surface.blit(title_img, self.title_pos)

        for i in range(len(self.parameters)):
            p_text_img = self._text_img(self.parameters[i], self.text_color)
            self.surface.blit(p_text_img, [self.p_x[i], self.p_name_y])
            v_text_img = self._text_img(f"{values[i]:.3f}", self.text_color)
            self.surface.blit(v_text_img, [self.p_x[i], self.p_value_y])

        self.parent_surface.blit(self.surface, self.position)

    def _text_img(self,
                  text: str,
                  text_color: Tuple[int, int, int]) -> pg.SurfaceType:
        return self.font.render(text, True, text_color, self.bg_color)


class AssistantEventHandler(EventHandler):
    """Event handler that includes assistant actions """

    ACTION_DT = 0.05
    """Action delta. The amount action offset is changed per
    key press. This is a proportion of the max range.
    """

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

    @classmethod
    def handle_event(cls,
                     action_type: ActionType,
                     event: pg.event.EventType) -> None:
        """ Overrides parent """
        if isinstance(action_type, AssistantContinuousAction):
            cls.handle_assistant_continuous_action_event(action_type, event)
        else:
            super().handle_event(action_type, event)

    @classmethod
    def handle_assistant_continuous_action_event(
            cls,
            action_type: AssistantContinuousAction,
            event: pg.event.EventType) -> None:
        """Event handler for Assistant Continuous Actions """
        action = action_type.last_assistant_action.copy()
        a_idxs = AssistantContinuousAction.ASSISTANT_ACTION_INDICES
        if event.type == pg.KEYDOWN and event.key in cls.KEY_ACTION_MAP:
            action_key, increase = cls.KEY_ACTION_MAP[event.key]
            action_idx = a_idxs[action_key]
            action[action_idx] += cls.action_dt(
                action[action_idx], increase
            )
        elif (
                event.type == pg.KEYUP
                and event.key in [pg.K_UP, pg.K_DOWN]
        ):
            # Reset steering to 0 upon key release
            action_key, _ = cls.KEY_ACTION_MAP[event.key]
            action_idx = a_idxs[action_key]
            action[action_idx] = 0.0
        action_type.act(action)

    @classmethod
    def action_dt(cls,
                  old_action: float,
                  increase: bool,
                  action_min: float = -1.0,
                  action_max: float = 1.0) -> float:
        """Get action offset step size """
        if increase:
            return min(abs(action_max - old_action), cls.ACTION_DT)
        return -min(abs(old_action - action_min), cls.ACTION_DT)
