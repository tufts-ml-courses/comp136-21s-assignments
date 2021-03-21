import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import logsumexp
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from RandomWalkSampler import RandomWalkSampler2D

def calc_model_log_pdf(w, logsigma, x_N, t_N):
    ''' Compute log pdf of all modeled random variables

    Args
    ----
    w : scalar float
    logsigma : scalar float
    x_N : 1D array, shape (N,)
    t_N : 1D array, shape (N,)

    Returns
    -------
    logpdf : float real scalar
        Log probability density function value at provided input
        Will be the *joint* probability: \log p(w, logsigma, t_1:N | x_1:N)
    '''
    log_pdf_t = 0.0 # TODO compute log p(t | x, w, logsigma)

    log_pdf_w = 0.0 # TODO compute log p(w)
    
    log_pdf_logsigma = 0.0 # TODO compute log p(logsigma)

    return log_pdf_t + log_pdf_w + log_pdf_logsigma


if __name__ == '__main__':
    n_samples = 40000
    n_keep = 30000
    random_state = 42
    prng = np.random.RandomState(random_state)

    N_grid = np.asarray([0, 1, 4, 512])
    G = len(N_grid)

    train_df = pd.read_csv("../data/toyline_train.csv")
    x_N = train_df['x'].values
    t_N = train_df['y'].values

    z_initA_D = np.asarray([0.02, 0.02])
    z_initB_D = np.asarray([1.0, -1.0])

    H = 3 # height of one panel
    W = 2 # width of one panel
    _, ax_grid = plt.subplots(
        nrows=2, ncols=G, sharex=True, sharey=True,
        figsize=(W*G, H*2))

    # TODO for each value of N
    # Run two separate MCMC chains initialized from 'A' and 'B' above
    # Make plot

    for col_id, N in enumerate(N_grid):
        # TODO actually run the sampler here
        zA_SD = prng.randn(n_keep, 2)  # fixme
        zB_SD = prng.randn(n_keep, 2)  # fixme

        # Make the plots
        ax_grid[0, col_id].plot(zA_SD[:,0], zA_SD[:,1], '.', alpha=0.05, color='r')
        ax_grid[1, col_id].plot(zB_SD[:,0], zB_SD[:,1], '.', alpha=0.05, color='b')

        ax_grid[0,col_id].set_title('N=%d' % N)
        ax_grid[1,col_id].set_xlabel('w')
        if col_id == 0:
            ax_grid[0,col_id].set_ylabel('$\\log \\sigma$')
            ax_grid[1,col_id].set_ylabel('$\\log \\sigma$')

    # Make plots pretty and standardized
    for ax in ax_grid.flatten():
        ax.set_xlim([-3, 3]);
        ax.set_ylim([-7, 5]);
        ax.set_aspect('equal', 'box');
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-6, -4, -2, 0, 2, 4])
    plt.tight_layout()
    plt.show()