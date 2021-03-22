import os.path as osp
from PIL import Image

KEY_BINDING_FIG_PATH = KEYBINDING_FIG_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "exercise_assistant_key_bindings.png"
)

EXPLANATION_FIG_PATH = KEYBINDING_FIG_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "exercise_assistant_explanation.png"
)


def display_keybindings():
    """Display keybinding figure in seperate window """
    img = Image.open(KEY_BINDING_FIG_PATH)
    img.show()


def display_explanation():
    """Display keybinding figure in seperate window """
    img = Image.open(EXPLANATION_FIG_PATH)
    img.show()
