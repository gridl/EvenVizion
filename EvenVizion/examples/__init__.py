"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion%20-%20video%20based%20camera%20localization%20component.pdf

This is licensed under an MIT license. See the LICENSE.md file
for more information.

This is an example of the comparison of visualizations between the original video and stabilized video.
"""

__version__ = "0.9"

__all__ = ['compare_evenvizion_with_original_video', 'evenvizion_component']

from .evenvizion_component import *
from .compare_evenvizion_with_original_video import *

