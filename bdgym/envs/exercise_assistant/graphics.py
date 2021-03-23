"""Functions and classes for rendering the Exercise Assistant Environment """
import os.path as osp
from typing import TYPE_CHECKING, Tuple, Optional

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


class EnvViewer:
    """A viewer to render a Exercise Assistant Environment """

    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (255, 225, 0)
    BLACK = (60, 60, 60)
    WHITE = (255, 255, 255)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN
    BG_COLOR = PURPLE

    def __init__(self,
                 env: 'ExerciseAssistantEnv',
                 screen_width: int = 900,
                 screen_height: int = 400,
                 save_images: bool = False,
                 save_directory: Optional[str] = None) -> None:
        self.env = env
        pg.init()
        pg.display.set_caption("Exercise Assistant Env")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_size = [screen_width, screen_height]
        self.screen = pg.display.set_mode(self.panel_size)
        self.sim_surface = pg.Surface(self.panel_size, 0)
        self.clock = pg.time.Clock()
        self.save_images = save_images
        self.save_directory = save_directory
        self.frame = 0

        self.assistant_graphics = AssistantGraphics(
            self.sim_surface,
            self.env.discrete_assistant,
            self.env.render_assistant_info
        )
        self.fig_graphics = FigureAnimationGraphics(self.sim_surface)
        self.athlete_graphics = AthleteGraphics(
            self.sim_surface, self.env.render_athlete_info
        )

    def display(self) -> None:
        """Update the pygame display of the environment """
        self.sim_surface.fill(self.BG_COLOR)

        if not self.env.is_athletes_turn():
            if self.env.athlete_performed_rep():
                self._perform_rep()
                return
            if self.env.set_count > 0:
                self._perform_end_set()
                return

        self._draw()
        pg.display.flip()

    def close(self) -> None:
        """Close the pygame window """
        pg.quit()

    def _draw(self, animate: bool = False) -> None:
        as_obs = self.env.last_assistant_obs
        if self.env.discrete_assistant:
            self.assistant_graphics.display(
                as_obs[0], self.env.assistant_offset
            )
        else:
            self.assistant_graphics.display(as_obs[0])

        end_set = (
            self.env.set_count > 0
            and not self.env.is_athletes_turn()
            and not self.env.athlete_performed_rep()
        )

        overexerted = self.env.athlete_overexerted()

        self.fig_graphics.display(
            self.env.set_count, animate, end_set, overexerted
        )
        self.athlete_graphics.display(self.env.last_athlete_obs)
        self.screen.blit(self.sim_surface, (0, 0))

        if self.save_images and self.save_directory:
            frame_num_str = self._get_frame_num_str()
            pg.image.save(
                self.sim_surface,
                osp.join(self.save_directory, f"frame_{frame_num_str}.png")
            )
            self.frame += 1

    def _perform_rep(self) -> None:
        self.fig_graphics.reset()
        for _ in range(self.fig_graphics.animation_length):
            self._draw(True)
            pg.display.update()
            self.clock.tick(self.fig_graphics.FRAME_RATE)

    def _perform_end_set(self) -> None:
        self.fig_graphics.reset()
        for _ in range(self.fig_graphics.animation_length):
            self._draw(False)
            pg.display.update()
            self.clock.tick(self.fig_graphics.FRAME_RATE)

    def _draw_overexerted(self) -> None:
        self.fig_graphics.reset()
        for _ in range(self.fig_graphics.animation_length):
            self._draw(False)
            pg.display.update()
            self.clock.tick(self.fig_graphics.FRAME_RATE)

    def _get_frame_num_str(self) -> str:
        if self.frame < 10:
            prefix_len = 3
        elif self.frame < 100:
            prefix_len = 2
        elif self.frame < 1000:
            prefix_len = 1
        else:
            prefix_len = 0
        return f"{'0' * prefix_len}{self.frame}"


class FigureAnimationGraphics:
    """Visualization of stick figure performing exercises """

    BG_COLOR = EnvViewer.WHITE
    OVEREXERTED_COLOR = EnvViewer.RED
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
            int(self.width/2 - self._set_count_text(0, False).get_width()/2),
            int(0.05*self.height)
        )

        # End Set Text
        self.end_set_text_pos = (
            int(self.width/2 - self._end_set_text(0).get_width()/2),
            int(0.2*self.height)
        )

    def display(self,
                set_count: int,
                animate: bool = False,
                end_set: bool = False,
                overexerted: bool = False) -> None:
        """Update Figure Animation Display """
        if overexerted:
            self.surface.fill(self.OVEREXERTED_COLOR)
        else:
            self.surface.fill(self.BG_COLOR)

        text_img = self._set_count_text(set_count, overexerted)
        self.surface.blit(text_img, self.text_pos)

        if overexerted:
            overexerted_text_img = self._overexerted_text()
            self.surface.blit(overexerted_text_img, self.end_set_text_pos)
        elif end_set:
            end_set_text_img = self._end_set_text(set_count-1)
            self.surface.blit(end_set_text_img, self.end_set_text_pos)

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

    def _set_count_text(self,
                        set_count: int,
                        overexerted: bool) -> pg.SurfaceType:
        text = f"Set {set_count}"
        if overexerted:
            bg_color = self.OVEREXERTED_COLOR
        else:
            bg_color = self.BG_COLOR
        return self.font.render(text, True, self.TEXT_COLOR, bg_color)

    def _end_set_text(self, set_count: int) -> pg.SurfaceType:
        text = f"Set {set_count} Ended"
        return self.font.render(text, True, EnvViewer.GREEN, self.BG_COLOR)

    def _overexerted_text(self) -> pg.SurfaceType:
        text = "Overexerted :("
        return self.font.render(
            text, True, EnvViewer.BLACK, self.OVEREXERTED_COLOR
        )

    @property
    def animation_length(self) -> int:
        """Get length of the animation """
        return len(self.animation_imgs) + 1


class AgentGraphics:
    """Base class for agent graphics """

    FONT_SIZE = 40
    TEXT_COLOR = EnvViewer.BLACK

    def __init__(self,
                 root_surface: pg.SurfaceType,
                 root_position: Tuple[int, int],
                 agent_name: str,
                 bg_color: Tuple[int, int, int] = EnvViewer.BG_COLOR):
        self.width = root_surface.get_width() // 3
        self.height = root_surface.get_height()
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)
        self.root_position = root_position
        self.root_surface = root_surface

        self.name = agent_name
        self.bg_color = bg_color

        # Text
        self.title_graphic = TextGraphic(
            self.surface,
            width=self.width,
            height=0.1*self.height,
            position=(0, int(0.05*self.height)),
            text=self.name,
            center_text=True,
            font_size=self.FONT_SIZE,
            bg_color=self.bg_color,
            text_color=self.TEXT_COLOR
        )

    def display(self, *args, **kwargs) -> None:
        """Update Athlete Graphic """
        self.surface.fill(self.bg_color)
        self.title_graphic.display()
        self._agent_specific_display(*args, **kwargs)
        self.root_surface.blit(self.surface, self.root_position)

    def _agent_specific_display(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    def bar_width(self) -> float:
        """Width of bars in agent display """
        return 0.8*self.width

    @property
    def bar_height(self) -> float:
        """Height of bars in agent display """
        return 0.1*self.height

    @property
    def bar_title_height(self) -> float:
        """Height of bar title in agent display """
        return 0.2*self.height

    @property
    def bar_gap(self) -> float:
        """Gap between bars in agent display """
        return 0.15*self.height

    @property
    def first_bar_pos(self) -> Tuple[int, int]:
        """Position of the first bar in agent display """
        return (0.1*self.width, 0.25*self.height)

    def get_bar_pos(self, bar_num: int) -> Tuple[int, int]:
        """Get position of bar """
        init_width, init_height = self.first_bar_pos
        return (
            init_width,
            init_height + bar_num*self.bar_gap + bar_num*self.bar_height
        )

    def get_bar_title_position(self, bar_num: int) -> Tuple[int, int]:
        """Height of bars in agent display """
        bar_x, bar_y = self.get_bar_pos(bar_num)
        return (bar_x, bar_y-(0.05*self.height))


class AssistantGraphics(AgentGraphics):
    """Visualization of the Assistants graphics """

    def __init__(self,
                 root_surface: pg.SurfaceType,
                 display_offset: bool,
                 render_assistant_info: bool):
        super().__init__(
            root_surface,
            root_position=(0, 0),
            agent_name="Assistant",
            bg_color=EnvViewer.BLUE
        )
        self.render_assistant_info = render_assistant_info
        self.display_offset = display_offset

        # Energy Bar
        if self.render_assistant_info:
            self.energy_title = TextGraphic(
                self.surface,
                width=None,
                height=None,
                position=self.get_bar_title_position(0),
                text="Athlete Energy",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )
            energy_bar_pos = self.get_bar_pos(0)
            self.energy_bar = BarGraphic(
                self.surface,
                width=self.bar_width,
                height=self.bar_height,
                position=energy_bar_pos,
                min_value=0.0,
                max_value=1.0,
                bg_color=self.bg_color,
                bar_fill_color=EnvViewer.GREEN
            )

            self.energy_value_text = TextGraphic(
                self.surface,
                width=self.bar_width,
                height=None,
                position=(
                    energy_bar_pos[0],
                    energy_bar_pos[1]+self.bar_height+0.01*self.height
                ),
                text="0.000",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )

        # Signal Offset Bar
        if self.render_assistant_info and self.display_offset:
            self.offset_title = TextGraphic(
                self.surface,
                width=None,
                height=None,
                position=self.get_bar_title_position(1),
                text="Applied Deception",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )
            offset_bar_pos = self.get_bar_pos(1)
            self.offset_bar = OffsetBarGraphic(
                self.surface,
                width=self.bar_width,
                height=self.bar_height,
                position=offset_bar_pos,
                min_value=-1.0,
                max_value=1.0,
                bg_color=self.bg_color,
                bar_fill_color=EnvViewer.GREEN
            )
            self.offset_value_text = TextGraphic(
                self.surface,
                width=self.bar_width,
                height=None,
                position=(
                    offset_bar_pos[0],
                    offset_bar_pos[1]+self.bar_height+0.01*self.height
                ),
                text="0.000",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )
        else:
            self.offset_bar = None

    def _agent_specific_display(self,
                                energy_obs: float,
                                offset_obs: float = None) -> None:
        if not self.render_assistant_info:
            return
        self.energy_title.display()
        self.energy_bar.display(energy_obs)
        self.energy_value_text.text = f"{energy_obs:.3f}"
        self.energy_value_text.display()
        if self.display_offset:
            self.offset_title.display()
            self.offset_bar.display(offset_obs)
            self.offset_value_text.text = f"{offset_obs:.3f}"
            self.offset_value_text.display()


class AthleteGraphics(AgentGraphics):
    """Visualization of the Athlete graphics """

    def __init__(self,
                 root_surface: pg.SurfaceType,
                 render_athlete_info: bool):
        super().__init__(
            root_surface,
            root_position=((2*root_surface.get_width() // 3), 0),
            agent_name="Athlete",
            bg_color=EnvViewer.YELLOW
        )
        self.render_athlete_info = render_athlete_info

        if self.render_athlete_info:
            # Percieved Energy Bar
            self.percieved_energy_title = TextGraphic(
                self.surface,
                width=None,
                height=None,
                position=self.get_bar_title_position(0),
                text="Percieved Energy",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )
            percieved_bar_pos = self.get_bar_pos(0)
            self.percieved_energy_bar = BarGraphic(
                self.surface,
                width=self.bar_width,
                height=self.bar_height,
                position=percieved_bar_pos,
                min_value=0.0,
                max_value=1.0,
                bg_color=self.bg_color,
                bar_fill_color=EnvViewer.GREEN
            )
            self.percieved_energy_value = TextGraphic(
                self.surface,
                width=self.bar_width,
                height=None,
                position=(
                    percieved_bar_pos[0],
                    percieved_bar_pos[1]+self.bar_height+0.01*self.height
                ),
                text="0.000",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )

        # Assistant Signal Energy Bar
        self.assistant_energy_title = TextGraphic(
            self.surface,
            width=None,
            height=None,
            position=self.get_bar_title_position(1),
            text="Assistant's Energy Signal",
            center_text=False,
            font_size=25,
            bg_color=self.bg_color,
            text_color=EnvViewer.BLACK
        )
        assistant_bar_pos = self.get_bar_pos(1)
        self.assistant_energy_bar = BarGraphic(
            self.surface,
            width=self.bar_width,
            height=self.bar_height,
            position=assistant_bar_pos,
            min_value=0.0,
            max_value=1.0,
            bg_color=self.bg_color,
            bar_fill_color=EnvViewer.GREEN
        )
        self.assistant_energy_value = TextGraphic(
                self.surface,
                width=self.bar_width,
                height=None,
                position=(
                    assistant_bar_pos[0],
                    assistant_bar_pos[1]+self.bar_height+0.01*self.height
                ),
                text="0.000",
                center_text=False,
                font_size=25,
                bg_color=self.bg_color,
                text_color=EnvViewer.BLACK
            )

        # Assistant Recommendation Bar
        self.assistant_rcmd_title = TextGraphic(
            self.surface,
            width=None,
            height=None,
            position=(0.1*self.width, 0.7*self.height),
            text="Assistant Recommendation:",
            center_text=False,
            font_size=25,
            bg_color=self.bg_color,
            text_color=EnvViewer.BLACK
        )
        self.assistant_rcmd_action = TextGraphic(
            self.surface,
            width=None,
            height=None,
            position=(0.1*self.width, 0.75*self.height),
            text="PERFORM_REP",
            center_text=False,
            font_size=30,
            bg_color=self.bg_color,
            text_color=EnvViewer.GREEN
        )

    def _agent_specific_display(self, obs: np.ndarray) -> None:
        if self.render_athlete_info:
            self.percieved_energy_title.display()
            self.percieved_energy_bar.display(obs[0])
            self.percieved_energy_value.text = f"{obs[0]:.3f}"
            self.percieved_energy_value.display()

        self.assistant_energy_title.display()
        self.assistant_energy_bar.display(obs[2])
        self.assistant_energy_value.text = f"{obs[2]:.3f}"
        self.assistant_energy_value.display()

        self.assistant_rcmd_title.display()
        if obs[3] > 0.5:
            self.assistant_rcmd_action.text = "END SET"
        else:
            self.assistant_rcmd_action.text = "PERFORM REP"
        self.assistant_rcmd_action.display()


class BaseGraphic:
    """A base graphic class """

    def __init__(self,
                 parent_surface: pg.SurfaceType,
                 width: int,
                 height: int,
                 position: Tuple[int, int],
                 bg_color: Optional[Tuple[int, int, int]] = None):
        self.parent_surface = parent_surface
        self.width = width
        self.height = height
        self.position = position
        self.bg_color = bg_color
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)

    def display(self, *args, **kwargs) -> None:
        """Update the graphic display """
        raise NotImplementedError

    def _draw_border(self):
        bar_rect = self.surface.get_rect()
        pg.draw.rect(self.surface, EnvViewer.BLACK, bar_rect, 1)


class TextGraphic(BaseGraphic):
    """A Text graphic """

    def __init__(self,
                 parent_surface: pg.SurfaceType,
                 width: Optional[int],
                 height: Optional[int],
                 position: Tuple[int, int],
                 text: str,
                 center_text: bool = False,
                 display_border: bool = False,
                 font_size: int = AthleteGraphics.FONT_SIZE,
                 bg_color: Optional[Tuple[int, int, int]] = None,
                 text_color: Tuple[int, int, int] = EnvViewer.GREEN):
        self.center_text = center_text
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.font = pg.font.Font(None, self.font_size)
        self.display_border = display_border

        if width is None or height is None:
            self.bg_color = bg_color
            # set width and height to size of text img
            img_width, img_height = self.size()
            if width is None:
                width = img_width
            if height is None:
                height = img_height

        super().__init__(parent_surface, width, height, position, bg_color)

    def display(self, *args, **kwargs) -> None:
        if self.bg_color:
            self.surface.fill(self.bg_color)

        text_img = self.font.render(
            self.text, True, self.text_color, self.bg_color
        )

        if self.center_text:
            img_width, img_height = text_img.get_size()
            text_pos = (
                int(self.width/2 - img_width/2),
                int(self.height/2 - img_height/2)
            )
        else:
            text_pos = (0, 0)

        self.surface.blit(text_img, text_pos)

        if self.display_border:
            self._draw_border()

        self.parent_surface.blit(self.surface, self.position)

    def size(self) -> Tuple[int, int]:
        """Get the width and height of text """
        text_img = self.font.render(
            self.text, True, self.text_color, self.bg_color
        )
        return text_img.get_size()


class BarGraphic(BaseGraphic):
    """Bar graphics """

    def __init__(self,
                 parent_surface: pg.SurfaceType,
                 width: int,
                 height: int,
                 position: Tuple[int, int],
                 min_value: float = 0.0,
                 max_value: float = 1.0,
                 bg_color: Tuple[int, int, int] = EnvViewer.BG_COLOR,
                 bar_fill_color: Tuple[int, int, int] = EnvViewer.GREEN):
        super().__init__(
            parent_surface,
            width,
            height,
            position,
            bg_color
        )
        self.min_value = min_value
        self.max_value = max_value
        self.bar_fill_color = bar_fill_color

    def display(self, value: float, *args, **kwargs) -> None:
        """Update the Bar Graphic """
        self.surface.fill(self.bg_color)
        self._draw_fill_rect(value)
        self._draw_border()
        self.parent_surface.blit(self.surface, self.position)

    def _draw_fill_rect(self, value: float):
        fill_proportion = value / (self.max_value - self.min_value)
        rect_position = (0, 0)
        rect_size = (fill_proportion*self.width, self.height)
        fill_rect = (*rect_position, *rect_size)
        pg.draw.rect(self.surface, self.bar_fill_color, fill_rect, 0)


class OffsetBarGraphic(BarGraphic):
    """A visualization of a offset bar """

    def _draw_fill_rect(self, value: float):
        if value != 0.0:
            self._draw_offset_rect(value)
        self._draw_mid_line()

    def _draw_mid_line(self):
        rect = (self.width // 2, 0, 1, self.height)
        pg.draw.rect(self.surface, EnvViewer.BLACK, rect, 1)

    def _draw_offset_rect(self, value: float):
        fill_proportion = abs(value) / (self.max_value - self.min_value)
        rect_size = (fill_proportion*self.width, self.height)
        if value > 0.0:
            rect_position = (self.width / 2, 0)
        else:
            rect_position = (self.width / 2 - rect_size[0], 0)

        fill_rect = (*rect_position, *rect_size)
        pg.draw.rect(self.surface, self.bar_fill_color, fill_rect, 0)
