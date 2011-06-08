import os
from os import path
import pytave

_M_FILES = path.abspath(path.join(__path__, "..", "octave"))
pytave.addpath(_M_FILES)

