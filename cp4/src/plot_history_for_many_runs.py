'''
plot_history_for_many_runs.py

Useful starter code for reading in CSV files and making plots

'''

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.25)

if __name__ == '__main__':
    for results_dir in ["results_many_LBFGS_runs/", "results_many_EM_runs/"]:
        K_list = [1, 4, 8, 16]
        seed_list = [1001, 3001, 4001]
        fig, ax_grid = plt.subplots(nrows=2, ncols=len(K_list), sharex=True, sharey=True, squeeze=False)
        for k, K in enumerate(K_list):
            for seed in seed_list:
                df = pd.read_csv(os.path.join(results_dir, 'history_K=%02d_seed=%04d.csv' % (K, seed)))
                ax_grid[0,k].plot(df['iter'], df['train_loss_per_pixel'], '.-', label='seed %d' % seed)
                ax_grid[1,k].plot(df['iter'], df['valid_neg_log_lik_per_pixel'], '.-', label='seed %d' % seed)

            ax_grid[0,0].set_ylabel("train loss / pixel")
            ax_grid[1,0].set_ylabel("valid neg log lik. / pixel")
            ax_grid[0,k].set_title("K=%d" % K)
        plt.show()