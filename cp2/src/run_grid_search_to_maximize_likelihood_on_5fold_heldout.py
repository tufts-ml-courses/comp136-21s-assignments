'''
Summary
-------
1. Select best hyperparameters (alpha, beta) of linear regression via a grid search
-- Use the LIKELIHOOD function of MAPEstimator on heldout set (average across K=5 folds).
2. Plot the best likelihood found vs. polynomial feature order.
-- Normalize scale of reported probabilities by dividing by the number of observations N
3. Report test set performance of best overall model (alpha, beta, order)
4. Report overall time required for model selection

'''

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import time


from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator

if __name__ == '__main__':
    train_df = pd.read_csv("../data/toywave_train.csv")
    test_df = pd.read_csv("../data/toywave_test.csv")

    x_train_ND, t_train_N = train_df['x'].values[:,np.newaxis], train_df['y'].values
    x_test_ND, t_test_N = test_df['x'].values[:,np.newaxis], test_df['y'].values

    order_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Coarse list of possible alpha
    # Finer list of possible beta (likelihoods matter more)
    params_to_search = dict(
        alpha=np.logspace(-4, 2, 7).tolist(),
        beta=np.logspace(-2, 2, 5 + 4 * 4).tolist(),
        )
    print("Possible alpha parameters")
    print(', '.join(['%.3f' % a for a in params_to_search['alpha']]))
    print("Possible beta parameters")
    print(', '.join(['%.3f' % a for a in params_to_search['beta']]))

    for N, line_color in [ (16, 'g'), (512, 'b')]:
        print("\n === Grid search for best (alpha, beta, order) on N=%d training set" % N)
        score_per_order = list()
        estimator_per_order = list()
        start_time = time.time()
        for order in order_list:
            feature_transformer = PolynomialFeatureTransform(order=order, input_dim=1)

            estimator = LinearRegressionMAPEstimator(feature_transformer)
            kfold_splitter = sklearn.model_selection.KFold(
                n_splits=5, shuffle=True, random_state=101)
            ## Create grid searcher object that will use estimator's score function
            ## TODO make sure you understand what these kwargs do!
            kfold_grid_searcher = sklearn.model_selection.GridSearchCV(
                estimator, params_to_search,
                cv=kfold_splitter, scoring=None, refit=True, return_train_score=True)
            ## TODO call fit on kfold_grid_searcher with the first N training points


            ## Select best scoring parameters
            best_estimator = estimator ## TODO replace this line! ask kfold_grid_searcher for best estimator
            best_score = 0.0 ## TODO replace this line! ask kfold_grid_searcher for best score averaged across folds
            print("order = %d | alpha = %9.3g beta = %9.3g | score % 9.3f" % (
                order, best_estimator.alpha, best_estimator.beta, best_score))
            estimator_per_order.append(best_estimator)
            score_per_order.append(best_score)

        plt.plot(order_list, score_per_order, 's-', label='N=%d' % N, color=line_color)

        # Add small vertical bar to indicate the order with *best* performance
        zs = np.zeros(2)
        ys = np.asarray([-0.05, +0.05])
        best_id = np.argmax(score_per_order)
        plt.plot(zs + order_list[best_id], ys + score_per_order[best_id], '--', color=line_color)
        
        # Report best performance of the best estimator 
        best_estimator_overall = estimator_per_order[best_id]

        # Write out required summary info 
        print("Best Overall MAPEstimator")
        print("order = %d" % order_list[best_id])
        print("alpha = %.3g" % best_estimator_overall.alpha)
        print("beta = %.3g" % best_estimator_overall.beta)
        print("test likelihood score: % 9.7f" % (
            best_estimator_overall.score(x_test_ND, t_test_N) / t_test_N.size))
        print("required time = %.2f sec" % (time.time() - start_time))

    ## Finalize plot
    plt.xlabel('polynomial order')
    plt.ylabel('heldout log lik. (avg over 5 folds)') 
    plt.legend(loc='lower right')       
    plt.ylim([-1.4, 0.1]) # Please keep these limits
    plt.show()

        