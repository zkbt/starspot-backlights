import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy, pkg_resources, os, glob, warnings

import astropy.units as u, astropy.constants as con


from scipy.interpolate import interp1d
from scipy.special import loggamma, gamma

data_directory = pkg_resources.resource_filename("starspotbacklights", "data")
