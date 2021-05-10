'''
Cartoon tool to model the TLSE.
'''

from .imports import *

import emcee, corner
from tqdm import tqdm

# this is needed to read PHOENIX model spectra
from rainbowconnection.sources.phoenix import *

# some numerical constants (avoiding astropy just for speed)
sigma_sb = 5.670367e-05 # ergs/s/cm**2/K**4
h = 6.6260755e-27	 # erg s
c = 2.99792458e10    # cm/s
k_B = 1.38e-16 # erg/K

Rearth = 6378100.0 # m
Rsun = 695700000.0 # m

# what is the maximum allowed covering fraction f
maxf = 0.5

# are we enforcing that spots must be dark?
darkspots = False

# some plotting defaults
linekw = dict(alpha=1)
markerkw = dict(alpha=1, s=1)

#labels = 'T_unspot', 'T_spot', 'f', 'deltaf', 'rp/rs'
parameter_names = 'T_unspot', 'T_spot', 'f', 'deltaf', 'f1', 'rprs'

def guesstimate_deltaf(f, f1=0.001):
    '''
    Make a Poisson-based estimate of the spot coverage.

    Parameters
    ----------
    f : float
        The average spot-covering fraction on the star.
    f1 : float
        The average spot-covering fraction of a single spot.

    Returns
    -------
    deltaf : float
        The estimate of deltaf, from assuming f1*sqrt(N)
    '''

    # make sure f is positive
    assert(np.all(f >= 0))

    # make sure it's symmetrical about 0.5 (this is a kludge!)
    deltaf = np.minimum(np.sqrt(f*f1), np.sqrt((1-f)*f1))

    return deltaf

def calculate_A(ratio, f, deltaf):
    '''
    Given a starspot brightness ratio, total spot covering fraction,
    and deltaf, estimate the amplitude of photometric variations.

    Parameters
    ----------
    ratio : float, np.array
        (spotted spectrum)/(unspotted spectrum)
    f : float
        The average spot-covering fraction on the star.
    deltaf : float
        The time-variable part of the spot-covering fraction.
    '''

    amplitude = np.abs(deltaf*(1-ratio)/(1 - f*(1-ratio)))
    return amplitude

def calculate_eps(ratio, f):
    '''
    Estimate the transit depth multiplicative factor,
    given some spot ratio and covering fraction, and
    assuming that f_tra (the spot-covering fraction along
    the transit chord) is 0.

    Parameters
    ----------
    ratio : float, np.array
        (spotted spectrum)/(unspotted spectrum)
    f : float
        The average spot-covering fraction on the star.

    Returns
    -------
    epsilon:
        The factor by which a transit depth is amplified.
        The observed transit depth will be epsilon*(Rp/R*)**2
    '''
    epsilon = 1/(1-f*(1-ratio))
    return epsilon

def solve_for_f(ratio=0.5, A=0.01, f1=0.001, visualize=False):
    '''
    A toy numerical solution for the total spot-covering fraction,
    given some ratio and amplitude, as well as and a strong belief
    on spot size for the Poisson-based deltaf.
    '''

    # create a grid of f values
    f = np.linspace(0, maxf, 1000)

    # create a grid of Poisson estimates for deltaf
    deltaf = guesstimate_deltaf(f, f1)

    # create a grid of amplitudes from those estimates
    A_hypothesis = calculate_A(ratio, f, deltaf)

    # find the best value of f
    best = np.argmin(np.abs(A_hypothesis - A))

    # make a plot showing this numerical solution
    if visualize:
        plt.plot(f, A_hypothesis)
        plt.scatter(f[best], A_hypothesis[best], color='black', zorder=10)
        plt.axhline(A)
        plt.axvline(f[best])
        plt.xlabel('$f$ (total spot covering fraction)')
        plt.ylabel('A = $\Delta I/I$')
        plt.title(f'ratio={ratio}, A={A}, f1={f1} | f={f[best]:.2f}')

    return f[best]

class Backlight:

    '''
    An object representing a

    '''
    def __init__(self, data={},
                       max_temperature_offset=0.3,
                       stellar_radius=1,
                       planet_radius=1,
                       logg=5.0,
                       include_poisson=True,
                       R=100,
                       extend_wavelengths=True):
        '''
        Parameters
        ----------

        data : pandas DataFrame
            Measurements of the system. Keys include...
                'oot' for amplitude of out-of-transit
                'depth' for transit depths
                'teff' for effective temperature of star
        max_temperature_offset : float
            By what fraction can the starspot temperature differ
            from the effective temperature of the star?
        stellar_radius : float
            What is the radius of star in solar radii?
        planet_radius : float
            What is the radius of planet in Earth radii?
        logg : float
            What is the logg of the star?
        include_poisson : bool
            Should we include a Poisson prior connecting
            f to a probability distribution for deltaf?
        R : float
            The resolution at which spectra should be loaded/plotted
        extend_wavelengths : bool
            Should we extend the wavelengths of the PHOENIX spectra?
        '''

        # the data used to constrain the backlight
        self.set_data(data)

        # how far can the starspot temperature different from the effective?
        self.max_temperature_offset = max_temperature_offset

        # stellar and planetary radius (for depths)
        self.stellar_radius = stellar_radius
        self.planet_radius = planet_radius

        # stellar logg (for spectral models) [FIXME = add Z]
        self.logg = logg

        # should we apply a poisson prior connecting f to deltaf?
        self.include_poisson = include_poisson

        # store the resolution at which we'll load spectra
        self.R = R
        self.extend_wavelengths = extend_wavelengths

        self.previous_T_spot = None
        self.previous_T_unspot = None

    def set_data(self, data):
        '''
        Connect a set of data to the model.

        Parameters
        ----------
        data : pandas DataFrame
            Measurements of the system. Keys include...
                'oot' for amplitude of out-of-transit
                'depth' for transit depths
                'teff' for effective temperature of star
        '''
        self.data = data

    def filelabel(self):
        '''
        Give a label to files associated with this backlight model.
        '''
        f = 'backlight-model-'
        f += '-'.join([f'{len(self.data[k])}{k}' for k in self.data])
        return f


    def set_parameters(self, parameters):
        '''
        Set the active parameter values for this backlight.
        '''
        self.T_unspot, self.T_spot, self.f, self.deltaf, self.f1, self.rprs = parameters
        self.parameters = parameters

        # set the spectra associated with the spotted and unspotted photospheres
        # (only load them if they need to change)
        if self.T_unspot != self.previous_T_unspot:
            self.w_unspot, self.f_unspot = read_phoenix(self.T_unspot, R=self.R, logg=self.logg, extend_wavelengths=self.extend_wavelengths)
            self.previous_T_unspot = self.T_unspot
        if self.T_spot != self.previous_T_spot:
            self.w_spot, self.f_spot = read_phoenix(self.T_spot, R=self.R, logg=self.logg, extend_wavelengths=self.extend_wavelengths)
            self.previous_T_spot = self.T_spot

    def lnprior(self, parameters, *args):
        '''
        Define the prior for modeling the TLSE.

        Parameters
        ----------
        parameters : np.array
            The parameters of the model, in this order:
            'T_unspot', 'T_spot', 'f', 'deltaf', 'f1', 'rprs'
        '''


        # pull out the individual parameters
        self.set_parameters(parameters)

        # do we enforce that spots must be dark?
        if darkspots:
            if self.T_spot/self.T_unspot > 1:
                return -np.inf

        # make sure the spots fall within the PHOENIX models
        if (self.T_unspot < 2300) | (self.T_unspot > 12000):
            return -np.inf
        if (self.T_spot < 2300) | (self.T_spot > 12000):
            return -np.inf

        # make sure the spot covering fraction is reasonable
        if (self.f < 0) | (self.f > maxf):
            return -np.inf

        # make sure that deltaf doesn't exceed f
        if (self.deltaf < 0) | (self.deltaf > self.f):
            return -np.inf

        # make sure that deltaf doesn't exceed f
        if (self.f1 < 0) | (self.f1 > self.deltaf):
            return -np.inf

        # impose a temperature prior if we know one
        if self.max_temperature_offset is not None:
            T_eff = (self.f*self.T_spot**4 + (1-self.f)*self.T_unspot**4)**0.25
            offset = self.T_spot/T_eff - 1
            if np.abs(offset) > self.max_temperature_offset:
                return -np.inf

        # start with a prior of 0
        lnprior = 0.0

        # add a prior on deltaf given f
        if self.include_poisson:
            # what's the expectation value for the number of spots facing us?
            N_expected = self.f/self.f1

            # what's an actual value of the number of spots facing us?
            # (use -deltaf to handle both f=0 and f=0.5 well)
            N = (self.f-self.deltaf)/self.f1

            # calculate Poisson probability
            poisson_prior = N*np.log(N_expected) - N_expected - loggamma(N+1)
            # the use of the gamma function extends this to
            # beyond just integer values of N

            if np.isfinite(poisson_prior):
                lnprior += poisson_prior
            else:
                lnprior = -np.inf
            # (this was the old kludge)
            #deltaf_center = guesstimate_deltaf(f, self.f1)
            #deltaf_width = deltaf_center*0.5 # KLUDGE
            #prior = -0.5*((deltaf - deltaf_center)**2/deltaf_width**2)

        return lnprior



    def lnlike(self, parameters):
        f'''
        Define the likelihood.

        Parameters
        ----------
        parameters : np.array
            The parameters of the model, in this order:
            'T_unspot', 'T_spot', 'f', 'deltaf', 'f1', 'rprs'
        '''

        # set the parameters to be active
        self.set_parameters(parameters)

        lnlike = 0.0

        if 'oot' in self.data:
            # use the amplitudes of oot as data points
            oot_data = self.data['oot']

            # calculate the flux ratios for each out-of-transit
            ratios = np.array([f.integrate(self.w_spot, self.f_spot)/f.integrate(self.w_unspot, self.f_unspot) for f in oot_data['filter']])

            # calculate the out-of-transit amplitudes
            amplitudes = calculate_A(ratios, self.f, self.deltaf)

            # calculate a chi-sq term for this subset
            oot_chisq = np.sum((oot_data['amplitude'] -  amplitudes)**2/oot_data['amplitude-error']**2)
            lnlike += -0.5*oot_chisq

        if 'teff' in self.data:
            # use the overall effective temperature as another data point
            teff_data = self.data['teff']

            effective_flux = sigma_sb*teff_data['teff']**4
            error_effective_flux = 4*teff_data['teff-error']/teff_data['teff']*effective_flux

            # calculate the total flux
            w_cm = self.w_unspot/1e7
            integral_spot = np.trapz(h*c/self.w_spot*1e7*self.f_spot, self.w_spot)
            integral_phot = np.trapz(h*c/self.w_unspot*1e7*self.f_unspot, self.w_unspot)
            total_flux = integral_spot*self.f + integral_phot*(1-self.f)

            # calculate a chi-sq term for this subset
            teff_chisq = np.sum((total_flux - effective_flux)**2/error_effective_flux**2)
            lnlike += -0.5*teff_chisq

        if 'depth' in self.data:
            # use the transit depths as data points
            depth_data = self.data['depth']
            ratios = np.array([f.integrate(self.w_spot, self.f_spot)/f.integrate(self.w_unspot, self.f_unspot) for f in depth_data['filter']])
            eps = calculate_eps(ratios, self.f)
            raw_depth = self.rprs**2
            model_depths = raw_depth*eps
            depth_chisq = np.sum((depth_data['depth'] - model_depths)**2/depth_data['depth-error']**2)
            lnlike += -0.5*depth_chisq

        return lnlike

    def lnprob(self, parameters):
        f'''
        Define the posterior probability.

        Parameters
        ----------
        parameters : np.array
            The parameters of the model, in this order:
            'T_unspot', 'T_spot', 'f', 'deltaf', 'f1', 'rprs'

        Returns
        -------
        lnprob, lnprior, lnlike
            Everything after the first will be stored in an emcee blob.
        '''

        # calculate the prior probability
        lnprior = self.lnprior(parameters)

        # only if the prior is finite, calculate the likelihood
        if np.isfinite(lnprior):
            lnlike = self.lnlike(parameters)
        else:
            lnlike = 0
        return lnprior + lnlike, lnprior, lnlike


    def rprs_actual(self):
        '''
        FIXME, do we still need this?
        '''
        return (self.planet_radius*Rearth)/(self.stellar_radius*Rsun)

    def sample(self, nburn=5000, nsteps=5000, nwalk=30, remake=False):
        '''
        Run an MCMC to sample from the probability distribution
        for TLSE parameters.

        Parameters
        ----------
        nburn : int
            The number of steps to skip.
        nsteps : int
            The number of steps after burn-in.
        nwalk : int
            The number of walkers.
        remake : bool
            Should we redo the sampling, even if a saved version exists?
        '''

        # define the filename for these samples
        filename = self.filelabel()+f'-{nburn}burn-{nsteps}steps-{nwalk}walkers.npy'

        # load the samples if possible, otherwise recreate them
        try:
            self.samples, = np.load(filename, allow_pickle=True)
            print(f'loaded samples from {filename}')
            assert(remake == False)
        except (IOError, AssertionError):
            print('generating MCMC samples; this may take a moment')

            # set the initial parameters
            ndim = len(parameter_names)
            if 'teff' in self.data:
                teff = self.data['teff']['teff'][0]
                if self.max_temperature_offset is None:
                    nudge = 100
                else:
                    nudge = teff*self.max_temperature_offset
                tmin = np.maximum(teff - nudge, 2300)
                tmax = np.minimum(teff + nudge, 12000)

            initial_T_unspot = np.random.uniform(tmin, tmax, nwalk)
            if darkspots:
                initial_T_spot = np.random.uniform(tmin, initial_T_unspot, nwalk)
            else:
                initial_T_spot = np.random.uniform(tmin, tmax, nwalk)
            initial_f =  np.random.uniform(0, maxf, nwalk)
            initial_deltaf = np.random.uniform(0, initial_f, nwalk)
            initial_f1 = np.random.uniform(0, initial_deltaf, nwalk)
            initial_rprs = np.random.uniform( self.rprs_actual()*0.9,  self.rprs_actual()*1.1, nwalk)

            p0 = np.transpose([initial_T_unspot,
                               initial_T_spot,
                               initial_f,
                               initial_deltaf,
                               initial_f1,
                               initial_rprs])

            # set up some emcee blobs to hang onto lnlike, lnprior, lnposterior
            #dtype = [("lnprior", float), ("lnlike", float)]

            # create the sampler and run it
            self.sampler = emcee.EnsembleSampler(nwalk, ndim, self.lnprob)#, blobs_dtype=dtype)
            for i in self.sampler.sample(p0, iterations=nburn + nsteps, progress=True):
                pass


            blobs = self.sampler.get_blobs()

            self.samples = {}
            self.samples['lnprior'] = blobs[nburn:,:,0].reshape(nwalk * nsteps)
            self.samples['lnlike'] =  blobs[nburn:,:,1].reshape(nwalk * nsteps)
            self.samples['parameters'] = self.sampler.chain[:, nburn:, :].reshape(nwalk * nsteps, ndim)


            np.save(filename, [self.samples])
            print(f'saved samples to {filename}')



        self.parameters_maxprob = self.samples['parameters'][np.argmax(self.samples['lnprior'] + self.samples['lnlike']), :]
        for i, x in enumerate(self.parameters_maxprob):
            print(parameter_names[i], '=', x)

        self.print_parameter_summary()

        self.define_subset()

    def print_parameter_summary(self):
        '''
        Print a summary of the parameters from the samples.
        '''
        def confidence_interval(x):
            one_sigma = 0.682689
            lower, middle, upper = np.percentile(x, [(1 - one_sigma)/2, 0.5, 1-(1 - one_sigma)/2], )
            ms, ls, us = f'{middle:.5}', f'{lower-middle:+.5}', f'{upper-middle:+.5}'
            return '{'+ms+'}'+'{'+ls+'}'+'{'+us+'}'

        print()
        for i, k in enumerate(parameter_names):
            print(f"{k} = {confidence_interval(self.samples['parameters'][:,i])}")

        for k in ['lnprior', 'lnlike']:
            print(f"{k} = {confidence_interval(self.samples[k])}")

    def plot_samples(self):
        '''
        Simple wrapper to make a corner plot of the samples.
        '''
        # Plot the corner plot
        fig = corner.corner(self.samples['parameters'], labels=parameter_names);

    def define_subset(self, N=10, seed=42):
        '''
        Define a random subset of the parameter set.
        '''
        np.random.seed(seed)
        self.subset_indices = np.random.randint(len(self.samples['parameters']), size=N)
        self.subset = self.samples['parameters'][self.subset_indices]

    def dye(self, parameters):
        '''
        Dye parameters based on their values,
        coloring each sample by its starspot
        temperature ratio.

        Parameters
        ----------
        parameters : np.array
            An array of a whole bunch of parameters.
        '''

        T_unspot, T_spot, f, deltaf, f1, rprs = parameters
        return np.log(T_spot/T_unspot)

    def setup_colors(self, factor=1.1):
        '''
        Set the shared color scale for all plots
        '''
        self.vmin, self.vmax = -np.log(factor),  np.log(factor)
        self.norm = plt.Normalize(self.vmin, self.vmax)
        self.cmap = plt.cm.coolwarm_r


    def plot_amplitudes(self, factor=1.1, **kw):
        '''
        Plot the (measured and) model wavelength-dependent variability amplitudes.
        '''

        self.setup_colors(factor=factor)

        # plot the data
        if 'oot' in self.data:
            data = self.data['oot']
            plt.errorbar(data['center'], 100*data['amplitude'], 100*data['amplitude-error'],
                         marker='.', markersize=10,  zorder=10, color='black',
                         linewidth=0, elinewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Semiamplitude of\nPhotometric Modulations (%)')
        plt.xlim(400, 2500)

        # plot a subsample of parameters
        for parameters in self.subset:

            self.set_parameters(parameters)
            A = 100*calculate_A(self.f_spot/self.f_unspot, self.f, self.deltaf)
            thisdye = self.dye(parameters)
            plt.plot(self.w_unspot, A, color=self.cmap(self.norm(thisdye)), **linekw)


    def plot_depth(self, factor=1.1, **kw):
        '''
        Plot the (measured and) model transit depths.
        '''

        self.setup_colors(factor=factor)

        # plot the data
        if 'depth' in self.data:
            data = self.data['depth']
            plt.errorbar(data['center'], 100*data['depth'], 100*data['depth-error'],
                         marker='.', markersize=10,  zorder=10, color='black',
                         linewidth=0, elinewidth=2)
        plt.xlabel('Wavelength (nm)')

        # plot a subsample of parameters

        for parameters in self.subset:

            self.set_parameters(parameters)
            eps = calculate_eps(self.f_spot/self.f_unspot, self.f)
            depth = 100*eps*self.rprs**2
            thisdye = self.dye(parameters)

            plt.plot(self.w_unspot, depth, color=self.cmap(self.norm(thisdye)), **linekw)
        plt.ylabel('Absolute Spot-Affected\nTransit Depths (%)')

    def plot_dyed_samples(self, encompassing_gs_panel,
                                horizontal = ['deltaf'],
                                vertical = [ 'ratio',  'N', 'f', 'rprs'],#'T_unspot',
                                max_samples=1000,
                                factor=1.1, **kw):
        '''
        Plot the samples along some useful projections.

        Parameters
        ----------
        encompassing_gs_panel : gridspec
            The gridspec panel into which the plots should be made.
            (It will be split up into many subplots.)
        horizontal : list
            The names of the parameters to plot on the x-axis
        vertical : list
            The names of the parameters to plot on the x-axis
        '''

        self.setup_colors(factor=factor)

        labels=dict(deltaf='$\Delta f(t)$',
                    rprs='$R_p/R_s$',
                    T_unspot='$T_{unspot}$',
                    ratio='$T_{spot}/T_{unspot}$',
                    f='$f$',
                    f1='$f_{1}$',
                    N='$N = f/f_{1}$')

        # define the parameter clouds
        T_unspot, T_spot, f, deltaf, f1, rprs = self.samples['parameters'].T
        ratio = T_spot/T_unspot
        N = f/f1

        # sort the samples randomly (to avoid visual biases)
        s = np.argsort(np.random.random(len(T_spot)))

        # create a subset (to avoid visual overcrowding)
        ok = np.random.randint(0, len(T_spot), max_samples)


        # loop through rows
        gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(len(vertical),
                                                             len(horizontal),
                                                             subplot_spec=encompassing_gs_panel)


        self.ax_scatter = {}
        ax = None
        # loop through rows
        for i, k in enumerate(vertical):
            # loop through columns
            for j, l in enumerate(horizontal):
                # pull out the x and y values to plot
                x = locals()[l]
                y = locals()[k]

                # create and store an axes for this row/column
                ax = plt.subplot(gs[i,j], sharex=ax)
                self.ax_scatter[f'x={l},y={k}'] = ax

                # set the colors for the points
                thisdye = self.dye(self.samples['parameters'][s][ok].T)

                # scatter the points in this panel
                plt.scatter(x[s][ok], y[s][ok], c=thisdye, norm=self.norm, cmap=self.cmap, **markerkw)

                # adjust tick and axis labels
                if i != len(vertical)-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    plt.xlabel(labels[horizontal[j]])
                if j == 0:
                    plt.ylabel(labels[k])

                N_to_plot = [1, 10, 100]
                if (k == 'N'):
                    for N in N_to_plot:
                        plt.axhline(N, linestyle='--', color='black', **linekw)

                if (k == 'ratio'):
                    plt.axhline(1, linestyle='--', color='black', **linekw)

                if (k == 'T_unspot'):
                    if 'teff' in self.data:
                        teff_data = self.data['teff']
                        plt.axhline(teff_data['teff'][0], color='black', **linekw)
                        for direction in [-1, 1]:
                            plt.axhline(teff_data['teff'][0]+direction*teff_data['teff-error'][0], linestyle='--', color='black', **linekw)


                if (k == 'f') and (l == 'deltaf'):
                    for this_N in N_to_plot:
                        this_f = np.linspace(0, 0.5)
                        this_f1 = this_f/this_N
                        this_deltaf = np.sqrt(this_f1*this_f)
                        plt.plot(this_deltaf, this_f, linestyle='--', color='black', **linekw)

                # adjust limits
                if k == 'ratio':
                    if self.max_temperature_offset is not None:
                        plt.ylim(1-self.max_temperature_offset, 1+self.max_temperature_offset)
                if k == 'f':
                    plt.ylim(0, 0.5)
                if l == 'deltaf':
                    plt.xlim(0, 0.1)
                if l == 'f':
                    plt.xlim(0, 0.5)

    def plot_everything(self, factor=1.1, seed=42, **kw):
        #to_plot = ['f',  'rprs', 'T_unspot', 'ratio',  'deltaf']

        self.setup_colors(factor=factor)

        # pull out a subset of samples
        self.define_subset(10, seed=seed)

        # create a figure
        fi = plt.figure(figsize=(7, 5))


        gs = plt.matplotlib.gridspec.GridSpec(2, 2,
                                              width_ratios=[1.5, 1],
                                              wspace=0.4,
                                              bottom=0.1, top=0.95)


        # plot the oot amplitude, with the oot data
        self.ax_amp = plt.subplot(gs[0,0])
        self.plot_amplitudes(**kw)
        plt.setp(self.ax_amp.get_xticklabels(), visible=False)
        plt.xlabel(' ')
        if 'oot' in self.data:
            plt.ylim(0, 100*np.max(self.data['oot']['amplitude']*1.5))
        else:
            plt.ylim(0, 1.5)
        #plt.xscale('log')


        self.ax_depth = plt.subplot(gs[1,0], sharex=self.ax_amp)
        self.plot_depth(**kw)
        D = self.rprs_actual()**2*100
        plt.ylim(D*0.9, D*1.1)

        self.plot_dyed_samples(gs[:,1], **kw)

    def plot_both(self):

        fi, ax = plt.subplots(1, 2, figsize=(10, 3))

        plt.sca(ax[0])
        self.plot_amplitudes()

        plt.sca(ax[1])
        self.plot_transmission()

        plt.tight_depths()
        return ax

    def plot_epsilon(self):
        for T_unspot, T_spot, f, deltaf, rprs in self.subset:

            w_unspot, f_unspot = read_phoenix(T_unspot, R=100, logg=self.logg)
            w_spot, f_spot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = calculate_eps(f_spot/f_unspot, f)
            plt.plot(w_spot, eps, alpha=0.5)
        plt.ylabel('Absolute Spot-Affected\nTransit Depths (%)')

    def plot_wfc3(self):
        for T_unspot, T_spot, f, deltaf in self.subset:

            w_unspot, f_unspot = read_phoenix(T_unspot, R=100, logg=self.logg)
            w_spot, f_spot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = calculate_eps(f_spot/f_unspot, f)
            eps /= np.mean(eps[(w_spot > 1100)&(w_spot < 1300)])
            plt.plot(w_spot, eps, alpha=0.5)
        plt.ylabel('WFC3 Relative Spot-Induced\nTransit Depth Variations')
        plt.ylim(.9, 1.1)
        plt.xlim(1000, 1700)

    def plot_spectra(self):

        fi, ax = plt.subplots(1, 3, figsize=(10, 3))

        plt.sca(ax[0])
        self.plot_amplitudes()

        plt.sca(ax[1])
        self.plot_epsilon()

        plt.sca(ax[2])
        self.plot_wfc3()

        plt.tight_layout()

    def plot_transmission(self):
        planet_depth = ((self.planet_radius*Rearth)/(self.stellar_radius*Rsun))**2

        for T_unspot, T_spot, f, deltaf, rprs in self.subset:

            w_unspot, f_unspot = read_phoenix(T_unspot, R=100, logg=self.logg)
            w_spot, f_spot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = calculate_eps(f_spot/f_unspot, f)
            eps /= np.mean(eps[(w_spot > 1100)&(w_spot < 1300)])
            plt.plot(w_spot, eps*planet_depth*100, **linekw)
        plt.ylabel('Spurious Spot-Induced WFC3\nTransmission Spectrum (%)')
        plt.xlabel('Wavelength (nm)')
        plt.xlim(1000, 1700)
