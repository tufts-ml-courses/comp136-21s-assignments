'''
Summary
-------
Defines a maximum a-posterior estimator for unigrams

MAPEstimator supports a common API for unigram probability estimators:
* fit
* predict_proba
* score

Resources
---------
See COMP 136 course website for the complete problem description and all math details
'''

import numpy as np
from Vocabulary import Vocabulary


class MAPEstimator():
    """
    Maximum A-Posteriori Estimator for unigram probabiliies

    Attributes
    ----------
    vocab : Vocabulary object
    alpha : float, must be greater than or equal to one
            Defines concentration parameter of the symmetric Dirichlet prior

    Examples
    --------
    >>> word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
    >>> mapEst = MAPEstimator(Vocabulary(word_list), alpha=2.0)
    >>> mapEst.fit(word_list)
    >>> np.allclose(mapEst.predict_proba('dinosaur'), 3.0/7.0)
    True

    >>> mapEst.predict_proba('never_seen-before')
    Traceback (most recent call last):
    ...
    KeyError: 'Word never_seen-before not in the vocabulary'
    """

    def __init__(self, vocab, alpha=2.0):
        self.vocab = vocab
        self.alpha = float(alpha)

        # State that is adjusted by calls to 'fit'
        self.total_count = 0
        self.count_V = None

    def fit(self, word_list):
        ''' Fit this estimator to provided training data

        Args
        ----
        word_list : list of str
                Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * The 1D array count_V is set to the count of each vocabulary word
        * The integer total_count is set to the total length of the word list
        '''
        self.count_V = np.zeros(self.vocab.size)
        self.total_count = 0
        # update total_count
        self.total_count = len(word_list)
        # update the count_V array
        for w in word_list:
            self.count_V[self.vocab.get_word_id(w)] += 1

    def predict_proba(self, word):
        ''' Predict probability of a given unigram under this model

        Assumes this word is in the vocabulary

        Args
        ----
        word : string
                Known word that can be looked up in the vocabulary

        Returns
        -------
        proba : float between 0 and 1

        Throws
        ------
        ValueError if hyperparameters do not allow MAP estimation
        KeyError if the provided word is not in the vocabulary
        '''
        # adjust if condition to avoid cases where valid MAP does not exist
        id = self.vocab.get_word_id(word)
        if self.alpha <= 1:
            raise ValueError(
                "Hyperparameter alpha does not yield valid MAP estimate")
        # calculate MAP estimate of the provided word
        return (self.count_V[id] +self.alpha -1) / (self.vocab.size * (self.alpha - 1) + self.total_count)


    def score(self, word_list):
        ''' Compute the average log probability of words in provided list

        Args
        ----
        word_list : list of str
                Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        avg_log_proba : float between (-np.inf, 0.0)
        '''
        total_log_proba = 0.0
        for word in word_list:
            total_log_proba += np.log(self.predict_proba(word))
        return total_log_proba / len(word_list)
