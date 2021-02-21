"""Functions and classes for rendering the Exercise Assistant Environment """
import os.path as osp
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pygame as pg

if TYPE_CHECKING:
    from bdgym.envs.exercise_assistant.env import ExerciseAssistantEnv


RESOURCE_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "resources"
)

EA_FIG_PREFIX = "EA_Fig"
EA_FIG_EXT = ".png"


# TODO
# 1. Add labels to each bar
# 2. Add titles to each bar
# 3. Add event handling
# 4. Add text for action selection information

class EnvViewer:
    """A viewer to render a Exercise Assistant Environment """

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
                 env: 'ExerciseAssistantEnv',
                 screen_width: int = 900,
                 screen_height: int = 400) -> None:
        self.env = env
        pg.init()
        pg.display.set_caption("Exercise Assistant Env")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_size = [screen_width, screen_height]
        self.screen = pg.display.set_mode(self.panel_size)
        self.sim_surface = pg.Surface(self.panel_size, 0)
        self.clock = pg.time.Clock()

        self.assistant_graphics = AssistantGraphics(
            self.sim_surface,
            self.env.discrete_assistant
        )
        self.fig_graphics = FigureAnimationGraphics(self.sim_surface)
        self.athlete_graphics = AthleteGraphics(self.sim_surface)

    def display(self) -> None:
        """Update the pygame display of the environment """
        self.sim_surface.fill(self.BG_COLOR)

        if not self.env.is_athletes_turn() \
           and self.env.athlete_performed_rep():
            self._perform_rep()
            return

        self._draw()
        pg.display.flip()

    def _draw(self, animate: bool = False) -> None:
        as_obs = self.env.last_assistant_obs
        if self.env.discrete_assistant:
            self.assistant_graphics.display(
                as_obs[0], self.env.assistant_offset
            )
        else:
            self.assistant_graphics.display(as_obs[0])

        self.fig_graphics.display(self.env.set_count, animate)
        self.athlete_graphics.display(self.env.last_athlete_obs)
        self.screen.blit(self.sim_surface, (0, 0))

    def _perform_rep(self) -> None:
        self.fig_graphics.reset()
        for _ in range(self.fig_graphics.animation_length):
            self._draw(True)
            pg.display.update()
            self.clock.tick(self.fig_graphics.FRAME_RATE)


class FigureAnimationGraphics:
    """Visualization of stick figure performing exercises """

    BG_COLOR = EnvViewer.BLUE
    NUM_IMGS = 4
    FRAME_RATE = 15
    FONT_SIZE = 40
    TEXT_COLOR = EnvViewer.BLACK

    def __init__(self, root_surface: pg.SurfaceType):
        self.width = root_surface.get_width() // 3
        self.height = root_surface.get_height()
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)
        # position of figure animation surface within the root surface
        self.root_position = (root_surface.get_width() // 3, 0)
        self.root_surface = root_surface

        # For figure animation
        self.animation_imgs = [
            pg.image.load(
                osp.join(RESOURCE_DIR, f"{EA_FIG_PREFIX}{i}{EA_FIG_EXT}")
            ) for i in range(self.NUM_IMGS)
        ]
        self.anim_num = 0
        self.fig_origin = (
            int(self.width/2 - self.animation_imgs[0].get_width()/2),
            int(0.9*self.height - self.animation_imgs[0].get_height())
        )

        # For Set count text
        self.font = pg.font.Font(None, self.FONT_SIZE)
        self.text_pos = (
            int(self.width/2 - self._set_count_text(0).get_width()/2),
            int(0.05*self.height)
        )

    def display(self, set_count: int, animate: bool = False) -> None:
        """Update Figure Animation Display """
        self.surface.fill(self.BG_COLOR)

        text_img = self._set_count_text(set_count)
        self.surface.blit(text_img, self.text_pos)

        if animate:
            anim_img = self._next_img()
        else:
            anim_img = self.animation_imgs[0]
        self.surface.blit(anim_img, self.fig_origin)
        self.root_surface.blit(self.surface, self.root_position)

    def reset(self):
        """Reset figure to initial position """
        self.anim_num = 0

    def _next_img(self) -> pg.SurfaceType:
        anim_img = self.animation_imgs[self.anim_num]
        self.anim_num = (self.anim_num + 1) % len(self.animation_imgs)
        return anim_img

    def _set_count_text(self, set_count: int) -> pg.SurfaceType:
        text = f"Set {set_count}"
        return self.font.render(text, True, self.TEXT_COLOR, self.BG_COLOR)

    @property
    def animation_length(self) -> int:
        """Get length of the animation """
        return len(self.animation_imgs) + 1


class AssistantGraphics:
    """Visualization of the Assistants graphics """

    BG_COLOR = EnvViewer.RED
    FONT_SIZE = 40
    TEXT_COLOR = EnvViewer.BLACK

    def __init__(self,
                 root_surface: pg.SurfaceType,
                 display_offset: bool):
        self.width = root_surface.get_width() // 3
        self.height = root_surface.get_height()
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)
        # position of figure animation surface within the root surface
        self.root_position = (0, 0)
        self.root_surface = root_surface

        # Text
        self.font = pg.font.Font(None, self.FONT_SIZE)
        self.text_pos = (
            int(self.width/2 - self._name_text().get_width()/2),
            int(0.05*self.height)
        )

        # Energy Bar
        self.energy_bar = BarGraphic(
            self.surface,
            width=0.1*self.width,
            height=0.5*self.height,
            position=(0.2*self.width, 0.2*self.height),
            min_value=0.0,
            max_value=1.0,
            bg_color=self.BG_COLOR,
            bar_fill_color=EnvViewer.GREEN
        )

        # Signal Offset Bar
        self.display_offset = display_offset
        if self.display_offset:
            self.offset_bar = OffsetBarGraphic(
                self.surface,
                width=0.1*self.width,
                height=0.5*self.height,
                position=(0.5*self.width, 0.2*self.height),
                min_value=-1.0,
                max_value=1.0,
                bg_color=self.BG_COLOR,
                bar_fill_color=EnvViewer.GREEN
            )
        else:
            self.offset_bar = None

    def display(self, energy_obs: float, offset_obs: float = None) -> None:
        """Update Assistant Graphic """
        self.surface.fill(self.BG_COLOR)

        text_img = self._name_text()
        self.surface.blit(text_img, self.text_pos)

        self.energy_bar.display(energy_obs)
        if self.display_offset:
            self.offset_bar.display(offset_obs)
        self.root_surface.blit(self.surface, self.root_position)

    def _name_text(self) -> pg.SurfaceType:
        text = "Assistant"
        return self.font.render(text, True, self.TEXT_COLOR, self.BG_COLOR)


class AthleteGraphics:
    """Visualization of the Athlete graphics """

    BG_COLOR = EnvViewer.YELLOW
    FONT_SIZE = 40
    TEXT_COLOR = EnvViewer.BLACK

    def __init__(self, root_surface: pg.SurfaceType):
        self.width = root_surface.get_width() // 3
        self.height = root_surface.get_height()
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)
        # position of figure animation surface within the root surface
        self.root_position = ((2*root_surface.get_width() // 3), 0)
        self.root_surface = root_surface

        # Text
        self.font = pg.font.Font(None, self.FONT_SIZE)
        self.title_text_pos = (
            int(self.width/2 - self._name_text().get_width()/2),
            int(0.05*self.height)
        )

        bar_width = 0.1*self.width
        bar_height = 0.5*self.height
        first_bar_pos = (0.2*self.width, 0.2*self.height)
        bar_gap = 0.2*self.width
        # Percieved Energy Bar
        self.percieved_energy_bar = BarGraphic(
            self.surface,
            width=bar_width,
            height=bar_height,
            position=first_bar_pos,
            min_value=0.0,
            max_value=1.0,
            bg_color=self.BG_COLOR,
            bar_fill_color=EnvViewer.GREEN
        )

        # Assistant Signal Energy Bar
        self.assistant_energy_bar = BarGraphic(
            self.surface,
            width=bar_width,
            height=bar_height,
            position=(
                first_bar_pos[0] + bar_width + bar_gap,
                first_bar_pos[1]
            ),
            min_value=0.0,
            max_value=1.0,
            bg_color=self.BG_COLOR,
            bar_fill_color=EnvViewer.GREEN
        )

    def display(self, obs: np.ndarray) -> None:
        """Update Athlete Graphic """
        self.surface.fill(self.BG_COLOR)

        title_text_img = self._name_text()
        self.surface.blit(title_text_img, self.title_text_pos)

        self.percieved_energy_bar.display(obs[0])
        self.assistant_energy_bar.display(obs[2])
        self.root_surface.blit(self.surface, self.root_position)

    def _name_text(self) -> pg.SurfaceType:
        text = "Athlete"
        return self.font.render(text, True, self.TEXT_COLOR, self.BG_COLOR)


class BarGraphic:
    """A visualization of a progress bar """

    def __init__(self,
                 parent_surface: pg.SurfaceType,
                 width: int,
                 height: int,
                 position: Tuple[int, int],
                 min_value: float = 0.0,
                 max_value: float = 1.0,
                 bg_color: Tuple[int, int, int] = EnvViewer.BG_COLOR,
                 bar_fill_color: Tuple[int, int, int] = EnvViewer.GREEN):
        self.parent_surface = parent_surface
        self.width = width
        self.height = height
        self.position = position
        self.min_value = min_value
        self.max_value = max_value
        self.bg_color = bg_color
        self.bar_fill_color = bar_fill_color
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)

    def display(self, value: float) -> None:
        """Update the Bar Graphic """
        self.surface.fill(self.bg_color)
        self._draw_fill_rect(value)
        self._draw_border()
        self.parent_surface.blit(self.surface, self.position)

    def _draw_fill_rect(self, value: float):
        fill_proportion = value / (self.max_value - self.min_value)
        rect_position = (0, (1-fill_proportion)*self.height)
        rect_size = (self.width, fill_proportion*self.height)
        fill_rect = (*rect_position, *rect_size)
        pg.draw.rect(self.surface, self.bar_fill_color, fill_rect, 0)

    def _draw_border(self):
        bar_rect = self.surface.get_rect()
        pg.draw.rect(self.surface, EnvViewer.BLACK, bar_rect, 1)


class OffsetBarGraphic:
    """A visualization of a offset bar """

    def __init__(self,
                 parent_surface: pg.SurfaceType,
                 width: int,
                 height: int,
                 position: Tuple[int, int],
                 min_value: float = -1.0,
                 max_value: float = 1.0,
                 bg_color: Tuple[int, int, int] = EnvViewer.BG_COLOR,
                 bar_fill_color: Tuple[int, int, int] = EnvViewer.GREEN):
        self.parent_surface = parent_surface
        self.width = width
        self.height = height
        self.position = position
        self.min_value = min_value
        self.max_value = max_value
        self.bg_color = bg_color
        self.bar_fill_color = bar_fill_color
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)

    def display(self, value: float) -> None:
        """Update the Bar Graphic """
        self.surface.fill(self.bg_color)
        self._draw_fill_rect(value)
        self._draw_border()
        self.parent_surface.blit(self.surface, self.position)

    def _draw_fill_rect(self, value: float):
        if value == 0.0:
            rect = (0, self.height // 2, self.width, 1)
            pg.draw.rect(self.surface, EnvViewer.BLACK, rect, 1)
            return

        fill_proportion = abs(value) / (self.max_value - self.min_value)
        rect_size = (self.width, fill_proportion*self.height)
        if value > 0.0:
            rect_position = (0, self.height/2 - rect_size[1])
        else:
            rect_position = (0, self.height/2)

        fill_rect = (*rect_position, *rect_size)
        pg.draw.rect(self.surface, self.bar_fill_color, fill_rect, 0)

    def _draw_border(self):
        bar_rect = self.surface.get_rect()
        pg.draw.rect(self.surface, EnvViewer.BLACK, bar_rect, 1)
