import numpy as np
from MLEstimator import MLEstimator
from Vocabulary import Vocabulary
from MAPEstimator import MAPEstimator
from PosteriorPredictiveEstimator import *

# word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
# mle = MLEstimator(Vocabulary(word_list), unseen_proba=0.1)
# mle.fit(word_list)
# print(mle.predict_proba('dinosaur'))

# word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
# mapEst = MAPEstimator(Vocabulary(word_list), alpha=2.0)
# mapEst.fit(word_list)
# print(np.allclose(mapEst.predict_proba('dinosaur'), 3.0 / 7.0))

word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
ppe = PosteriorPredictiveEstimator(Vocabulary(word_list), alpha=2.0)
ppe.fit(word_list)
print(np.allclose(ppe.predict_proba('dinosaur'), 4.0 / 10.0))
