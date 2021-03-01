'''
Summary
-------
This script produces two figures exploring visual performance of estimators
* One using a likelihood estimator based on MAP estimated weights
* One using a posterior predictive estimator that marginalizes out weights
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator
from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator

if __name__ == '__main__':
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    order_list = [0, 1, 2, 3]
    n_train_list = [8, 512]
        
    alpha = 1.0 # moderate prior precision
    beta = 20.0 # strong likelihood precision

    for dataset_name in ['toyline', 'toywave']:
        # Load and unpack training and test data
        train_df = pd.read_csv("../data/%s_train.csv" % dataset_name)
        test_df = pd.read_csv("../data/%s_test.csv" % dataset_name)

        x_train_ND = train_df['x'].values[:,np.newaxis]
        t_train_N = train_df['y'].values
        x_test_ND = test_df['x'].values[:,np.newaxis]
        t_test_N = test_df['y'].values

        # To visualize prediction function learned from data,            
        # Create dense grid of G values between x.min() - R, x.max() + R
        # Basically, 3 times as wide as the observed data values for 'x'
        G = 201 # num grid points
        xmin = x_train_ND[:,0].min()
        xmax = x_train_ND[:,0].max()
        R = xmax - xmin
        xgrid_G = np.linspace(xmin - R, xmax + R, G)
        xgrid_G1 = np.reshape(xgrid_G, (G, 1))

        # Prepare figure 1 for MAP, figure 2 for PPE
        nrows = len(n_train_list)
        ncols = len(order_list)
        panel_size = 3
        fig1, fig1_map_axgrid = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex=True, sharey=True, squeeze=True,
            figsize=(panel_size * ncols, panel_size * nrows))
        fig2, fig2_ppe_axgrid = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex=True, sharey=True, squeeze=True,
            figsize=(panel_size * ncols, panel_size * nrows))

        # Loop over order of polynomial features
        # and associated axes of our two figures
        for order, fig1_map_ax, fig2_ppe_ax in zip(
                order_list, fig1_map_axgrid.T, fig2_ppe_axgrid.T):

            feature_transformer = PolynomialFeatureTransform(
                order=order, input_dim=1)

            # Allocate space for results
            map_train_scores = np.zeros(len(n_train_list))
            map_test_scores = np.zeros(len(n_train_list))
            ppe_train_scores = np.zeros(len(n_train_list))
            ppe_test_scores = np.zeros(len(n_train_list))

            print("===== MAPEstimator with order %d, alpha %.3g, beta %.3g" % (order, alpha, beta))
            for ii, N in enumerate(n_train_list):
                # Train MAP estimator using only first N examples
                map_estimator = LinearRegressionMAPEstimator(
                    feature_transformer, alpha=alpha, beta=beta)
                ## TODO fit estimator on first N examples in train

                # Evaluate heldout likelihood score on full train and test
                ## TODO record estimator's score on train in map_train_scores
                ## TODO record estimator's score on test in map_test_scores
                print("%6d examples : train score % 9.3f | test score % 9.3f" % (
                    N, map_train_scores[ii], map_test_scores[ii]))

                # Obtain predicted mean
                mean_G = np.zeros(G) # TODO call predict to get map_estimator's mean
                # Obtain predicted variance. 
                stddev_G = np.ones(G) # TODO fixme (Hint: depends on beta)

                # Plot the training data
                fig1_map_ax[ii].plot(x_test_ND[:100,0], t_test_N[:100],
                    'r.', label='test data', alpha=0.2)
                fig1_map_ax[ii].plot(x_train_ND[:N,0], t_train_N[:N],
                    'ks', markersize=4, label='train data', alpha=0.4, markeredgecolor=None)
                # Plot the predicted mean and the +/- 3 std dev interval
                fig1_map_ax[ii].fill_between(
                    xgrid_G, mean_G - 3 * stddev_G, mean_G + 3 * stddev_G,
                    facecolor='blue', alpha=0.5, label='3 stddev range')
                fig1_map_ax[ii].plot(
                    xgrid_G, mean_G, 'b-', label='mean prediction')
                # Show the test score in panel's title
                fig1_map_ax[ii].set_title("N=%d, order=%d: test lik %.2f" % (
                    n_train_list[ii], order, map_test_scores[ii]))
                # Make figure beautiful
                fig1_map_ax[ii].set_ylim([-5, 5]);
                fig1_map_ax[ii].set_yticks([-4, -2, 0, 2, 4]); 
                fig1_map_ax[ii].set_xlim([-5, 5])
                fig1_map_ax[ii].set_xticks([-4, -2, 0, 2, 4]); 
                fig1_map_ax[ii].set_aspect('equal', 'box')
                if ii == len(n_train_list) - 1:
                    fig1_map_ax[ii].set_xlabel("input $x$")
                if order == 0:
                    fig1_map_ax[ii].set_ylabel("predicted value $t$")
            if order == order_list[0]:
                fig1_map_ax[-1].legend(loc='upper left', fontsize=8)


            print("===== PosteriorPredictive with order %d, alpha %.3g, beta %.3g" % (order, alpha, beta))
            for ii, N in enumerate(n_train_list):
                # Train MAP estimator using only first N examples
                map_estimator = LinearRegressionPosteriorPredictiveEstimator(
                    feature_transformer, alpha=alpha, beta=beta)
                ## TODO fit estimator on first N examples in train

                # Evaluate heldout likelihood score on full train and test
                ## TODO record estimator's score on train in ppe_train_scores
                ## TODO record estimator's score on test in ppe_test_scores
                print("%6d examples : train score % 9.3f | test score % 9.3f" % (
                    N, ppe_train_scores[ii], ppe_test_scores[ii]))

                # Obtain predicted mean
                mean_G = np.zeros(G) # TODO call predict to get mean
                # Obtain predicted variance. 
                stddev_G = np.ones(G) # TODO call predict_variance

                # Plot the training data
                fig2_ppe_ax[ii].plot(x_test_ND[:100,0], t_test_N[:100],
                    'r.', label='test data', alpha=0.2)
                fig2_ppe_ax[ii].plot(x_train_ND[:N,0], t_train_N[:N],
                    'ks', markersize=4, label='train data', alpha=0.4, markeredgecolor=None)
                # Plot the predicted mean and the +/- 3 std dev interval
                fig2_ppe_ax[ii].fill_between(
                    xgrid_G, mean_G - 3 * stddev_G, mean_G + 3 * stddev_G,
                    facecolor='green', alpha=0.5, label='3 stddev range')
                fig2_ppe_ax[ii].plot(
                    xgrid_G, mean_G, 'g-', label='mean prediction')
                # Show the test score in panel's title
                fig2_ppe_ax[ii].set_title("N=%d, order=%d: test lik %.2f" % (
                    n_train_list[ii], order, map_test_scores[ii]))
                # Make figure beautiful
                fig2_ppe_ax[ii].set_ylim([-5, 5]);
                fig2_ppe_ax[ii].set_yticks([-4, -2, 0, 2, 4]); 
                fig2_ppe_ax[ii].set_xlim([-5, 5])
                fig2_ppe_ax[ii].set_xticks([-4, -2, 0, 2, 4]); 
                fig2_ppe_ax[ii].set_aspect('equal', 'box')
                if ii == len(n_train_list) - 1:
                    fig2_ppe_ax[ii].set_xlabel("input $x$")
                if order == 0:
                    fig2_ppe_ax[ii].set_ylabel("predicted value $t$")
            if order == order_list[0]:
                fig2_ppe_ax[-1].legend(loc='upper left', fontsize=8)            

        plt.figure(fig1.number)
        plt.suptitle(
            "MAP Predictions given alpha %.2f, beta %.2f" % (alpha, beta))

        plt.figure(fig2.number)
        plt.suptitle(
            "PPE Predictions given alpha %.2f, beta %.2f" % (alpha, beta))

    plt.show()
