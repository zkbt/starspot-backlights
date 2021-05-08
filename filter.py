import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

class Filter:
    """
    A filter is a function that can be integrated
    against some spectrum. Everything should be in nm.
    """

    def __init__(self, center=452, width=20):
        """
        Initialize with a center and a total width.
        """
        self.center = center
        self.width = width

    def __repr__(self):
        """
        How do we summarize this filter?
        """
        try:
            return "<{} ({:.0f}/{:.0f}nm) filter>".format(
                self.name, self.center, self.width
            )
        except:
            return "<{}/{}nm filter>".format(
                self.center, self.width
            )

    def __call__(self, w):
        """
        When the filter is called, return the response (= what
        fraction of photons are recorded at a particular wavelength?
        """
        response = np.zeros_like(w)
        inside = np.abs(w - self.center) < self.width
        response[inside] = 1.0
        return response

    def integrate(self, w, f, visualize=False):
        """
        Integrate this filter against a spectrum,
        and divide by the area of the filter response.
        """
        if visualize:
            plt.plot(w, f)
            plt.plot(w, self(w) * f)
            plt.xlim(
                self.center - 2 * self.width,
                self.center + 2 * self.width,
            )

        return np.trapz(f * self(w), w) / np.trapz(self(w), w)

    def calculate_cw(self):
        w = np.arange(300, 2500)
        self.center = np.trapz(w * self(w), w) / np.trapz(
            self(w), w
        )
        self.width = np.sqrt(
            np.trapz((w - self.center) ** 2 * self(w), w)
            / np.trapz(self(w), w)
        )


class MEarth(Filter):
    def __init__(self):
        angstrom, response = np.loadtxt("mearth-filter.txt").T
        self.model = interp1d(
            angstrom / 10,
            response,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self.name = "MEarth"
        self.calculate_cw()

    def __call__(self, w):
        return self.model(w)
