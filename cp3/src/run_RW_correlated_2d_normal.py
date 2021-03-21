'''
Purpose
-------
Sample from a 2-dim. Normal distribution using a Random Walk Sampler.
This is the Metropolis MCMC algorithm with a Gaussian proposal with controllable stddev

Target distribution:
# mean
>>> mu_D = np.asarray([-1.0, 1.0])
# covariance
>>> cov_DD = np.asarray([[2.0, 0.95], [0.95, 1.0]])

'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from RandomWalkSampler import RandomWalkSampler2D

def calc_target_log_pdf(z_D):
    ''' Compute log pdf of provided z value under target bivariate Normal distribution

    Args
    ----
    z_D : 1D array, size (D,)
        Value of the random variable at which we should compute log pdf

    Returns
    -------
    logpdf : float real scalar
        Log probability density function value at provided input
    '''
    ## TODO compute logpdf of z_D value under target distribution
    return -0.5 * np.sum(np.square(z_D)) # fixme


if __name__ == '__main__':
    n_samples = 10000   # total number of iterations of MCMC
    n_keep = 5000       # number samples to keep
    random_state = 42   # seed for random number generator
    prng = np.random.RandomState(random_state)

    # Two initializations, labeled 'A' and 'B'
    z_initA_D = np.zeros(2)
    z_initB_D = np.asarray([1.0, -1.0])

    # Look at a range of std. deviations
    rw_stddev_grid = np.asarray([0.01, 0.1, 1.0, 10.0])
    G = len(rw_stddev_grid)

    # Prepare a plot to view samples from two chains (A/B) side-by-side
    _, ax_grid = plt.subplots(
        nrows=2, ncols=G, sharex=True, sharey=True,
        figsize=(2*G, 2*2))

    for rr, rw_stddev in enumerate(rw_stddev_grid):

        # TODO Create samplers and run them for specified num iterations
        # Make sure to provide rw_stddev and random_state as args

        # TODO Stack list of samples into a 2D array of size (S, D)
        # Keeping only the last n_keep samples (and thus discarding burnin)
        zA_SD = prng.randn(n_keep, 2)  # fixme
        zB_SD = prng.randn(n_keep, 2)  # fixme

        # TODO unpack info about accept rates
        samplerA_accept_rate_last_half = 0.0
        samplerB_accept_rate_last_half = 0.0

        # Plot samples as scatterplot
        # Use small alpha transparency value for visual debugging of rare/frequent samples
        ax_grid[0,rr].set_title("rw_stddev = %.2F" % rw_stddev)
        ax_grid[0,rr].plot(zA_SD[:,0], zA_SD[:,1], 'r.', alpha=0.05)
        ax_grid[1,rr].plot(zB_SD[:,0], zB_SD[:,1], 'b.', alpha=0.05)
        # Mark initial points with "X"
        ax_grid[0,rr].plot(z_initA_D[0], z_initA_D[1], 'rx')
        ax_grid[1,rr].plot(z_initB_D[0], z_initB_D[1], 'bx')

        # Pretty print some stats for the samples
        # To give a way to check "convergence" from the terminal's stdout
        msg_pattern = ("RW stddev %5.2f from init %s | kept %d of %d samples | accept rate %.3f"
            + "\n    percentiles z0: 10th % 5.2f   50th % 5.2f   90th % 5.2f" 
            + "\n    percentiles z1: 10th % 5.2f   50th % 5.2f   90th % 5.2f"
            )
        print(msg_pattern % (
            rw_stddev, 'A', n_keep, n_samples,
            samplerA_accept_rate_last_half,
            *tuple(np.percentile(zA_SD[:,0:1], [10, 50, 90], axis=0)),
            *tuple(np.percentile(zA_SD[:,1:2], [10, 50, 90], axis=0)),
            ))
        print(msg_pattern % (
            rw_stddev, 'B', n_keep, n_samples,
            samplerB_accept_rate_last_half,
            *tuple(np.percentile(zB_SD[:,0:1], [10, 50, 90], axis=0)),
            *tuple(np.percentile(zB_SD[:,1:2], [10, 50, 90], axis=0)),
            ))

    # Make plots pretty and standardized
    for ax in ax_grid.flatten():
        ax.set_xlim([-5, 5]);
        ax.set_ylim([-5, 5]);
        ax.set_aspect('equal', 'box');
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])

    plt.tight_layout()
    plt.show()
