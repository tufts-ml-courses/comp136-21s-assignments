'''
Summary
-------
This script produces a figure showing how several estimators perform at the task of
computing the log probability of heldout words (y-axis) as training set size increases (x-axis).


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Vocabulary import Vocabulary
from MLEstimator import MLEstimator
from MAPEstimator import MAPEstimator
from PosteriorPredictiveEstimator import PosteriorPredictiveEstimator

if __name__ == '__main__':
    vocab = Vocabulary(["../data/training_data.txt", "../data/test_data.txt"])

    # Read in word list from plain-text file
    # The call to strip makes sure we have no words with lead/trailing whitespace
    train_word_list = [str.strip(s)
                       for s in np.loadtxt("../data/training_data.txt", dtype=str, delimiter=' ')]
    test_word_list = [str.strip(s)
                      for s in np.loadtxt("../data/test_data.txt", dtype=str, delimiter=' ')]

    frac_train_list = [1./128, 1./64, 1./32, 1./16, 1./8, 1./4, 1./2, 1.]
    n_train_list = [int(np.ceil(frac * len(train_word_list)))
                    for frac in frac_train_list]

    # Preallocate arrays to store the scores for each estimator
    mle_scores = -9.8 + np.zeros_like(frac_train_list)
    map_scores = -9.9 + np.zeros_like(frac_train_list)
    ppe_scores = -10 + np.zeros_like(frac_train_list)

    # TODO fit ML Estimator on train, then score it on the test set
    # TODO fit MAP Estimator on train, then score it on the test set
    # TODO fit PosteriorPredictive Estimator on train, then score it on the test set

    fig_h, ax_h = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(6, 4))
    arange_list = np.arange(len(frac_train_list))
    ax_h.plot(arange_list, mle_scores, 'm.-', label='ML estimator')
    ax_h.plot(arange_list, map_scores, 'b.-', label='MAP estimator')
    ax_h.plot(arange_list, ppe_scores, 'g.-', label='PosteriorPred estimator')

    ax_h.set_xticks([a for a in arange_list[::2]])
    ax_h.set_xticklabels(['%d' % a for a in n_train_list][::2])
    ax_h.set_xlim([0, max(arange_list)])
    ax_h.set_ylim([-10.1, -8.9])

    plt.legend(loc='lower right')
    plt.show()
