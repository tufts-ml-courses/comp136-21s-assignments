import numpy as np
import scipy.stats

class GibbsSampler2D(object):
    ''' Sampler for performing MCMC using Gibbs conditionals

    Requires a target random variable with 2-dimensional continuous real sample space.
    
    Assumes conditional distributions can be sampled from if provided a draw from a
    standard normal distribution, which can then be *transformed*. This allows the
    output of these functions to be *deterministic* at easily verifiable.

    Example Usage
    -------------
    # Pick a target distribution: 2-dim. standard normal
    # Then define ways to sample from each target condtional distribution
    # These methods take the other values of target r.v. and a std normal value
    >>> def draw_z0_given_z1(z1, u_stdnorm):
    ...     z0 = 1.0 * u_stdnorm + 0.0
    ...     return z0
    >>> def draw_z1_given_z0(z0, u_stdnorm):
    ...     z1 = 1.0 * u_stdnorm + 0.0
    ...     return z1

    # Create sampler
    >>> sampler = GibbsSampler2D(draw_z0_given_z1, draw_z1_given_z0, random_state=42)

    # Draw samples starting a specified initial value
    >>> z_D_list, info = sampler.draw_samples(zinit_D=np.zeros(2), n_samples=3000)

    # Use samples in to estimate mean of the distribution
    >>> np.mean(np.vstack(z_D_list), axis=0)
    array([-0.00689298,  0.00065241])

    Attributes
    ----------
    D : dimension (assumed 2 throughout)
    draw_z0_given_z1 : function
        Given a value of z1, produces sample for z0 (first dim of r.v. 1D array)
        Args:
        * z1 : scalar float
        Returns:
        * z0 : scalar float
    draw_z1_given_z0 : function
        Given a value of z0, produces sample for z1 (second dim of r.v. 1D array)
        Args:
        * z0 : scalar float
        Returns:
        * z1 : scalar float
    random_state : numpy.random.RandomState
        Pseudorandom number generator, supports .rand() and .randn()
    '''

    def __init__(self, draw_z0_given_z1, draw_z1_given_z0, random_state=0):
        ''' Constructor for GibbsSampler2D object for sampling from a 2D distribution

        User provides two functions to sample from required conditionals:
        * p(z0 | z1)
        * p(z1 | z0)

        Args
        ----
        draw_z0_given_z1 : function
            Given a value of z1, produces sample for z0 (first dim of r.v. 1D array)
            Args:
            * z1 : scalar float
            Returns:
            * z0 : scalar float
        draw_z1_given_z0 : function
            Given a value of z0, produces sample for z1 (second dim of r.v. 1D array)
            Args:
            * z0 : scalar float
            Returns:
            * z1 : scalar float
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
        self.draw_z1_given_z0 = draw_z1_given_z0
        self.draw_z0_given_z1 = draw_z0_given_z1
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

        # Repeat the same MCMC transition for many iterations/samples
        for s in range(n_samples):
            # Draw 'standard' random variables from this object's pseudo-random number generator
            u_D = self.sample_2D_standard_normal()

            # TODO construct a next sample of z_D given previous value z_D and randomness u_D
            # first call : ___ = self.draw_z0_given_z1(__, u_D[0])
            # second call: ___ = self.draw_z1_given_z0(__, u_D[1])
            znew_D = 1.0 * z_D + u_D # fixme

            # TODO make sure to add updated z_D to z_list
            z_list.append(z_D)

        # Record accept statistics and return
        sample_info = dict()
        did_accept_S = np.ones(n_samples, dtype=np.float64)
        sample_info['did_accept_S'] = did_accept_S
        sample_info['accept_rate'] = 1.0
        sample_info['accept_rate_last_half'] = np.mean(did_accept_S[-n_samples//2:])
        return z_list, sample_info