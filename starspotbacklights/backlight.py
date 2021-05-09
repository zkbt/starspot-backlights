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
Rearth = 6378100.0 # m
Rsun = 695700000.0 # m

# what is the maximum allowed covering fraction f
maxf = 0.5

# are we enforcing that spots must be dark?
darkspots = False

# some plotting defaults
linekw = dict(alpha=0.5)
labels = 'T_phot', 'T_spot', 'f', 'deltaf', 'rp/rs'

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

def estimate_A(ratio, f, deltaf):
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

def estimate_eps(ratio, f):
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
    A_hypothesis = estimate_A(ratio, f, deltaf)

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
    def __init__(self, data, max_temperature_offset=0.3, stellar_radius=1, planet_radius=1, f1=0.001, logg=5.0, include_poisson=True):
        '''
        Parameters
        ----------

        data : pandas DataFrame
            Measurements of the system.
            'oot' for amplitude of out-of-transit
            'depth' for transit depths
            'teff' for effective temperature of star
        stellar_radius : float
            Radius of star in solar radii
        planet_radius : float
            Radius of planet in Earth radii

        '''
        self.data = data
        self.max_temperature_offset = max_temperature_offset
        self.stellar_radius = stellar_radius
        self.planet_radius = planet_radius
        self.f1 = f1
        self.include_poisson = include_poisson
        self.logg = logg

    def filelabel(self):
        f = 'backlight-model-'
        f += '-'.join([f'{len(self.data[k])}{k}' for k in self.data])
        return f

    def lnprior(self, parameters, *args):
        '''
        Define the prior for modeling the TLSE.
        '''

        T_phot, T_spot, f, deltaf, rprs = parameters

        # do we enforce that spots must be dark?
        if darkspots:
            if T_spot/T_phot > 1:
                return -np.inf

        # make sure the spots fall within the PHOENIX models
        if (T_phot < 2300) | (T_phot > 12000):
            return -np.inf
        if (T_spot < 2300) | (T_spot > 12000):
            return -np.inf

        # make sure the spot covering fraction is reasonable
        if (f < 0) | (f > maxf):
            return -np.inf

        # make sure that deltaf doesn't exceed 50% (or f)
        if (deltaf < 0) | (deltaf > f):
            return -np.inf

        # add a prior on deltaf given f
        if self.include_poisson:
            deltaf_center = guesstimate_deltaf(f, self.f1)
            deltaf_width = deltaf_center*0.5 # KLUDGE
            prior = -0.5*((deltaf - deltaf_center)**2/deltaf_width**2)
        else:
            prior = 0.0

        # impose a temperature prior if we know one
        if self.max_temperature_offset is not None:
            T_eff = (f*T_spot**4 + (1-f)*T_phot**4)**0.25
            offset = T_spot/T_eff - 1
            if np.abs(offset) > self.max_temperature_offset:
                prior += -np.inf

        return prior

    def lnlike(self, parameters, data, visualize=True):
        '''
        Define the likelihood.
        '''

        T_phot, T_spot, f, deltaf, rprs = parameters

        # calculate the amplitudes of the photometric monitoring
        wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
        wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

        lnlike = 0.0

        if 'oot' in data:
            # use the amplitudes of oot as data points
            oot_data = data['oot']
            ratios = np.array([f.integrate(wspot, fspot)/f.integrate(wphot, fphot) for f in oot_data['filter']])
            amplitudes = estimate_A(ratios, f, deltaf, f1=self.f1)
            oot_chisq = np.sum(( oot_data['amplitude'] -  amplitudes)**2/oot_data['amplitude-error']**2)
            lnlike += -0.5*oot_chisq

        if 'teff' in data:
            # use the overall effective temperature as another data point
            teff_data = data['teff']
            effective_flux = sigma_sb*teff_data['teff']**4
            error_effective_flux = 4*teff_data['teff-error']/teff_data['teff']*effective_flux
            w_cm = wphot/1e7
            integral_spot = np.trapz(h*c/wspot*1e7*fspot, wspot)
            integral_phot = np.trapz(h*c/wphot*1e7*fphot, wphot)
            total_flux = integral_spot*f + integral_phot*(1-f)
            teff_chisq = np.sum((total_flux - effective_flux)**2/error_effective_flux**2)
            lnlike += -0.5*teff_chisq


        if 'depth' in data:
            # use the transit depths as data points
            depth_data = data['depth']
            ratios = np.array([f.integrate(wspot, fspot)/f.integrate(wphot, fphot) for f in depth_data['filter']])
            eps = estimate_eps(ratios, f)
            raw_depth = rprs**2
            model_depths = raw_depth*eps
            depth_chisq = np.sum((depth_data['depth'] - model_depths)**2/depth_data['depth-error']**2)
            lnlike += -0.5*depth_chisq

                # effective_flux = sigma_sb*teff_data['teff']**4
        # error_effective_flux = 4*teff_data['teff-error']/teff_data['teff']*effective_flux
        # w_cm = wphot/1e7
        # integral_spot = np.trapz(h*c/wspot*1e7*fspot, wspot)
        # integral_phot = np.trapz(h*c/wphot*1e7*fphot, wphot)
        # total_flux = integral_spot*f + integral_phot*(1-f)
        # effective_term = -0.5*(total_flux - effective_flux)**2/error_effective_flux**2

        return lnlike

    def lnprob(self, *args):
        '''
        Calculate the log(probability) for some parameters
        '''
        prob = self.lnprior(*args)
        if np.isfinite(prob):
            prob += self.lnlike(*args)
        return prob

    def rprs_actual(self):
        return (self.planet_radius*Rearth)/(self.stellar_radius*Rsun)

    def sample(self, nburn = 500, nsteps = 1000, nwalk = 30, remake=False):
        '''
        Run an MCMC to sample from the probability distribution
        for TLSE parameters.
        '''

        filename = self.filelabel()+f'-{nburn}burn-{nsteps}steps-{nwalk}walkers.npy'
        try:
            self.samples = np.load(filename)[()]
            print(f'loaded samples from {filename}')
            assert(remake == False)
        except (IOError, AssertionError):
            print('generating MCMC samples; this may take a moment')
            ndim = 5
            if 'teff' in self.data:
                tmin = np.maximum(self.data['teff']['teff'][0] - 100, 2300)
                tmax = np.minimum(self.data['teff']['teff'][0] + 100, 12000)
            else:
                tmin = 3000
                tmax = 10000
            rprs = self.rprs_actual()

            f = np.random.uniform(0, maxf, nwalk)
            if darkspots:
                tphot = np.random.uniform(tmin, tmax, nwalk)
                p0 = np.transpose([tphot,
                     np.random.uniform(tmin, tphot, nwalk),
                     f,
                     np.random.uniform(0, f, nwalk),
                     np.random.uniform(rprs*0.9, rprs*1.1, nwalk)])
            else:
                p0 = np.transpose([np.random.uniform(tmin, tmax, nwalk),
                      np.random.uniform(tmin, tmax, nwalk),
                      f,
                      np.random.uniform(0, f, nwalk),
                      np.random.uniform(rprs*0.9, rprs*1.1, nwalk)])

            self.sampler = emcee.EnsembleSampler(nwalk, ndim, self.lnprob,
                                    args=[self.data])
            for i in tqdm(self.sampler.sample(p0, iterations=nsteps), total=nsteps):
                pass

            self.maxlike = self.sampler.flatchain[np.argmax(self.sampler.flatlnprobability)]
            for i, x in enumerate(self.maxlike):
                print(labels[i], '=', x)

            self.samples = self.sampler.chain[:, nburn:, :].reshape(nwalk * (nsteps - nburn), ndim)
            np.save(filename, self.samples)
            print(f'saved samples to {filename}')

        self.define_subset()

    def plot_samples(self):
        # Plot the corner plot
        fig = corner.corner(self.samples, labels=labels);

    def define_subset(self, N=10, seed=42):
        np.random.seed(seed)
        self.subset = self.samples[np.random.randint(len(self.samples), size=N)]


    def dye(self, T_phot, T_spot, f, deltaf, rprs):
        return np.log(T_spot/T_phot)

    def setup_colors(self, factor=1.5):
        self.vmin, self.vmax = -np.log(factor),  np.log(factor)#np.percentile(T_spot/T_phot, [0, 100])
        self.norm = plt.Normalize(self.vmin, self.vmax)
        self.cmap = plt.cm.coolwarm_r


    def plot_amplitudes(self):
        '''
        Plot the wavelength-dependent amplitudes predicted by the samples.
        '''

        if 'oot' in self.data:
            data = self.data['oot']
            plt.errorbar(data['center'], 100*data['amplitude'], 100*data['amplitude-error'],
                         marker='.', markersize=10,  zorder=10, color='black',
                         linewidth=0, elinewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Semiamplitude of\nPhotometric Modulations (%)')
        plt.xlim(400, 2500)

        for T_phot, T_spot, f, deltaf, rprs in self.subset:

            wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
            wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

            A = 100*estimate_A(fspot/fphot, f, deltaf, f1=self.f1)

            thisdye = self.dye(T_phot, T_spot, f, deltaf, rprs)
            plt.plot(wphot, A, color=self.cmap(self.norm(thisdye)), **linekw)



    def plot_depth(self):

        if 'depth' in self.data:
            data = self.data['depth']
            plt.errorbar(data['center'], 100*data['depth'], 100*data['depth-error'],
                         marker='.', markersize=10,  zorder=10, color='black',
                         linewidth=0, elinewidth=2)
        plt.xlabel('Wavelength (nm)')

        for T_phot, T_spot, f, deltaf, rprs in self.subset:

            wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
            wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = estimate_eps(fspot/fphot, f)
            depth = 100*eps*rprs**2

            thisdye = self.dye(T_phot, T_spot, f, deltaf, rprs)
            plt.plot(wphot, depth, color=self.cmap(self.norm(thisdye)), **linekw)
        plt.ylabel('Absolute Spot-Affected\nTransit Depths (%)')

    def plot_dyed_samples(self, gs, toplot = ['deltaf', 'rprs', 'T_phot', 'ratio',  'f']):
        labels=dict(ratio='$T_{spot}/T_{unspot}$',
                    f='$f$', deltaf='$\Delta f(t)$',
                    T_phot='$T_{unspot}$',
                    rprs='$R_p/R_s$')


        T_phot, T_spot, f, deltaf, rprs = self.samples.T
        ratio = T_spot/T_phot
        s = np.argsort(np.random.random(len(T_spot)))
        ok = np.random.randint(0, len(T_spot), 10000)

        # loop through rows
        self.ax_scatter = {}
        ax = None
        for i, k in enumerate(toplot[1:]):
            # loop through columns
            for j, l in enumerate([toplot[0]]):
                x = locals()[l]
                y = locals()[k]

                ax = plt.subplot(gs[i,j+ 1], sharex=ax)
                self.ax_scatter[k] = ax
                thisdye = self.dye(T_phot[s][ok], T_spot[s][ok], f[s][ok], deltaf[s][ok], rprs[s][ok])
                plt.scatter(x[s][ok], y[s][ok], c=thisdye, norm=self.norm, cmap=self.cmap, s=1)
                if i != len(toplot)-2:
                    plt.setp(ax.get_xticklabels(), visible=False)
                plt.ylabel(labels[k])

                #if l == 'ratio':
                #    plt.xscale('log')
                if k == 'ratio':
                    if self.max_temperature_offset is not None:
                        plt.ylim(1-self.max_temperature_offset, 1+self.max_temperature_offset)

                if k == 'f':
                    plt.ylim(0, 0.5)


        plt.xlabel(labels[toplot[0]]);
        if l == 'deltaf':
            plt.xlim(0, 0.05)
        if l == 'f':
            plt.xlim(0, 0.5)

    def plot_everything(self, factor=1.2, seed=42):
        #toplot = ['f',  'rprs', 'T_phot', 'ratio',  'deltaf']

        self.setup_colors(factor=factor)

        # pull out a subset of samples
        self.define_subset(10, seed=seed)

        # create a figure
        fi = plt.figure(figsize=(7, 5))


        gs = plt.matplotlib.gridspec.GridSpec(4, 2,
                                              width_ratios=[2, 1],
                                              wspace=0.35,
                                              bottom=0.1, top=0.95)



        # plot the oot amplitude, with the oot data
        self.ax_amp = plt.subplot(gs[0:2,0])
        self.plot_amplitudes()
        plt.setp(self.ax_amp.get_xticklabels(), visible=False)
        plt.xlabel(' ')
        if 'oot' in self.data:
            plt.ylim(0, 100*np.max(self.data['oot']['amplitude']*1.5))
        else:
            plt.ylim(0, 1.5)
        #plt.xscale('log')


        self.ax_depth = plt.subplot(gs[2:4,0], sharex=self.ax_amp)
        self.plot_depth()
        D = self.rprs_actual()**2*100
        plt.ylim(D*0.9, D*1.1)

        self.plot_dyed_samples(gs)

    def plot_epsilon(self):
        for T_phot, T_spot, f, deltaf, rprs in self.subset:

            wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
            wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = estimate_eps(fspot/fphot, f)
            plt.plot(wspot, eps, alpha=0.5)
        plt.ylabel('Absolute Spot-Affected\nTransit Depths (%)')

    def plot_wfc3(self):
        for T_phot, T_spot, f, deltaf in self.subset:

            wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
            wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = estimate_eps(fspot/fphot, f)
            eps /= np.mean(eps[(wspot > 1100)&(wspot < 1300)])
            plt.plot(wspot, eps, alpha=0.5)
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

        for T_phot, T_spot, f, deltaf, rprs in self.subset:

            wphot, fphot = read_phoenix(T_phot, R=100, logg=self.logg)
            wspot, fspot = read_phoenix(T_spot, R=100, logg=self.logg)

            eps = estimate_eps(fspot/fphot, f)
            eps /= np.mean(eps[(wspot > 1100)&(wspot < 1300)])
            plt.plot(wspot, eps*planet_depth*100, **linekw)
        plt.ylabel('Spurious Spot-Induced WFC3\nTransmission Spectrum (%)')
        plt.xlabel('Wavelength (nm)')
        plt.xlim(1000, 1700)

    def plot_both(self):

        fi, ax = plt.subplots(1, 2, figsize=(10, 3))

        plt.sca(ax[0])
        self.plot_amplitudes()

        plt.sca(ax[1])
        self.plot_transmission()

        plt.tight_layout()
        return ax
