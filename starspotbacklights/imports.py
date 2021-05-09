import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy, pkg_resources, os, glob, warnings

import astropy.units as u, astropy.constants as con


from scipy.interpolate import interp1d

data_directory = pkg_resources.resource_filename("rainbowconnection", "data")
