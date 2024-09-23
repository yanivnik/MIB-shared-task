from typing import Optional, Literal

import numpy as np
import cmapy


EDGE_TYPE_COLORS = {
    'q': "#FF00FF", # Purple
    'k': "#00FF00", # Green
    'v': "#0000FF", # Blue
    None: "#000000", # Black
}

def get_color(qkv: Optional[Literal['q','k','v']], score:float):
    if qkv is not None:
        return EDGE_TYPE_COLORS[qkv]
    elif score < 0:
        return "#FF0000"
    else:
        return "#000000"

def rgb2hex(rgb):
    """
    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    return rgb2hex(cmapy.color(colorscheme, np.random.randint(0, 256), rgb_order=True))
    