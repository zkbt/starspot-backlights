from .imports import *

class Filter:
    """
    A filter is a function that can be integrated
    against some spectrum. Everything should be in nm.
    """

    def __init__(self, center=452, width=20):
        """
        Initialize with a center and a width,
        effectively creating a tophat filter
        spanning `center +/- width`.

        Parameters
        ----------
        center : float
            Central wavelength (in nm).
        width : float
            Half width (in nm), meaning that
            wavelengths within `width` of `center`,
            whether higher or lower, will be included.
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

        Parameters
        ----------
        w : np.array
            Wavelength, in nm

        Returns
        -------
        response : np.array
            The fractional response of the filter at
            each wavelength.
        """
        response = np.zeros_like(w)
        inside = np.abs(w - self.center) < self.width
        response[inside] = 1.0
        return response

    def integrate(self, w, f, visualize=False):
        """
        Integrate this filter against a spectrum,
        and divide by the area of the filter response.

        Parameters
        ----------
        w : np.array
            Wavelength, in nm

        f : np.array
            Flux, any units

        visualize : bool
            Should an explanatory plot be made?
        """
        if visualize:
            plt.plot(w, f)
            plt.plot(w, self(w) * f)
            plt.xlim(
                self.center - 2 * self.width,
                self.center + 2 * self.width,
            )

        return np.trapz(f * self(w), w) / np.trapz(self(w), w)

    def calculate_center_and_width(self):
        '''
        Calculate the center and effective width of the filter,
        and store these as attributes of the filter object.
        '''

        # FIX ME! make this more general!
        try:
            w = self.model.x
        except AttributeError:
            w = np.arange(300, 2500)
        self.center = np.trapz(w * self(w), w) / np.trapz(
            self(w), w
        )
        self.width = np.sqrt(
            np.trapz((w - self.center) ** 2 * self(w), w)
            / np.trapz(self(w), w)
        )

class MEarth(Filter):
    '''
    Define the MEarth filter bandpass.
    '''
    def __init__(self):
        '''
        Load the MEarth filter from a text file and set
        up an interpolation function using it.
        '''

        # load the filter data
        w_in_angstrom, response = np.loadtxt(os.path.join(data_directory, "mearth-filter.txt")).T
        w_in_nm = w_in_angstrom/10

        # set up the interpolation
        self.model = interp1d(
            w_in_nm,
            response,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self.name = "MEarth"
        self.calculate_center_and_width()

    def __call__(self, w):
        """
        When the filter is called, return the response (= what
        fraction of photons are recorded at a particular wavelength?

        Parameters
        ----------
        w : np.array
            Wavelength, in nm
        """
        return self.model(w)
