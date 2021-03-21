import numpy as np
import scipy.stats

class RandomWalkSampler2D(object):
    ''' Sampler for performing Metropolis MCMC with Random Walk proposals

    Requires a target random variable with 2-dimensional continuous real sample space.
    
    Example Usage
    -------------
    # Define a target distribution (2-dim. standard normal)
    >>> def calc_target_log_pdf(z_D):
    ...     logpdf1 = scipy.stats.norm.logpdf(z_D[0], 0, 1)
    ...     logpdf2 = scipy.stats.norm.logpdf(z_D[1], 0, 1)
    ...     return logpdf1 + logpdf2

    # Create sampler
    >>> sampler = RandomWalkSampler2D(calc_target_log_pdf, 0.5, random_state=42)

    # Draw samples starting a specified initial value
    >>> z_D_list, info = sampler.draw_samples(zinit_D=np.zeros(2), n_samples=3000)

    # Use samples in to estimate mean of the distribution
    >>> np.mean(np.vstack(z_D_list), axis=0)
    array([-0.18919171, -0.01949101])

    Attributes
    ----------
    D : dimension (assumed 2 throughout)
    calc_target_logpdf : function
        Given a value of random variable, computes the logpdf
        Args:
        * z_D, 1D array
        Returns:
        * logpdf : scalar float
            Log probability density of provided value of array z_D
            Can be accurate up to an additive constant
    rw_stddev : 1D array, size D
        Standard deviation of random walk proposal
    random_state : numpy.random.RandomState
        Pseudorandom number generator, supports .rand() and .randn()
    '''

    def __init__(self, calc_target_log_pdf, rw_stddev=1.0, random_state=0):
        ''' Constructor for RandomWalkSampler2D object

        Args
        ----
        calc_target_log_pdf : function
            Given a value of random variable, computes the logpdf
            Args:
            * z_D, 2D array
            Returns:
            * logpdf : scalar float
                Log probability density of provided sample
                Can be accurate up to an additive constant
        rw_stddev : scalar float
            Standard deviation of random walk proposal
        random_state : int or numpy.random.RandomState
            Initial state of this sampler's random number generator.
            Setting this deterministically enables reproducability and debugging.
            If integer, will create a numpy PRNG with that as seed.
            If numpy PRNG object, will call its randn() and rand() methods.

        Returns
        -------
        New RandomWalkSampler2D object
        '''
        self.D = 2
        self.calc_target_log_pdf = calc_target_log_pdf
        self.rw_stddev_D = float(rw_stddev) * np.ones(self.D, dtype=np.float64)

        if hasattr(random_state, 'rand'):
            self.random_state = random_state
        else:
            # Will raise error if not cast-able to int
            self.random_state = np.random.RandomState(int(random_state))

    def sample_2D_standard_normal(self):
        ''' Draw 2 values from "standard" Normal with mean 0.0 and variance 1.0

        Uses internal pseudo-random number generator to safely draw repeatable values.

        Args
        ----
        None.

        Returns
        -------
        u_D : 1D array, size (2,)
        '''
        return self.random_state.randn(self.D)

    def sample_uniform_0_to_1(self):
        ''' Draw scalar float from Uniform between 0.0 and 1.0

        Uses internal pseudo-random number generator to safely draw repeatable values.

        Args
        ----
        None.

        Returns
        -------
        rand_val : scalar float
        '''
        return self.random_state.rand(1)

    def draw_samples(self, zinit_D=None, n_samples=100):
        ''' Draw samples from target distribution via MCMC

        Args
        ----
        zinit_D : 1D array, size (D,) = (2,)
            Initial state of the target random variable to sample.
        n_samples : int
            Number of samples (iterations of MCMC).

        Returns
        -------
        z_list : list of numpy arrays
            Each entry is a sample from the MCMC procedure.
            Will contain n_samples+1 entries (includes initial state)
        sample_info : dict
            Contains information about this MCMC chain's progress, including
            'accept_rate' : Number of accepted proposals across all iterations
            'accept_rate_last_half' : Number of accepted proposals across last half of iterations
            'did_accept_S' : binary array of size S, indicated if each sample was accepted (1) or not (0)
        '''
        # Initialize with provided array
        z_D = 1.0 * zinit_D
        z_list = list()
        z_list.append(z_D)

        # Track whether each sample was accepted
        n_accept = 0
        did_accept_S = np.zeros(n_samples, dtype=np.float64)

        # Repeat the same MCMC transition for many iterations/samples
        for s in range(n_samples):
            # Draw 'standard' random variables from this object's pseudo-random number generator
            u_D = self.sample_2D_standard_normal()
            u_accept = self.sample_uniform_0_to_1()

            # TODO construct a proposed value zprime_D given previous value z_D and randomness u_D
            zprime_D = 1.0 * z_D # fixme

            # TODO compute Metropolis accept threshold A
            # using self.calc_target_log_pdf
            A = 1.0 # fixme

            if u_accept < A:
                # Accept!
                # TODO set z_D appropriately
                # TODO update n_accept and did_accept_S                
                pass

            else:
                # Reject!
                # TODO set z_D appropriately
                pass

            # TODO make sure to add updated z_D to z_list
            z_list.append(z_D)

        # Record accept statistics and return
        accept_rate_last_half = np.mean(did_accept_S[-n_samples//2:])
        sample_info = dict(
            n_accept=n_accept,
            n_samples=n_samples,
            accept_rate=n_accept/n_samples,
            accept_rate_last_half=accept_rate_last_half,
            did_accept_S=did_accept_S)
        return z_list, sample_info