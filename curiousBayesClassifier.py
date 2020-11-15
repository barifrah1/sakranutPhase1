import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from bayesClassifier import BayesClassifier
from tqdm import tqdm


class CuriousBayesClassifier(BayesClassifier):
    # constructor
    def __init__(self, data, columnsInfo):
        BayesClassifier.__init__(self, data, columnsInfo)
        self.indecies_of_data_rows = range(len(self.data))
        self.learning_order = []

    # get priors and posterior of specific row and compute IG

    def fit(self):
        self.D_KL = []
        iter = 1
        for row in tqdm(self.data):
            step_dkl = self.step(row)
            self.D_KL.append(step_dkl)
            if(iter % 10000 == 0):
                print('mean D_KL for iter ', iter, 'is: ', np.mean(self.D_KL))
            iter += 1
        return self.theta

    # calculate test error - for inside purposes only
    def calculate_test_error(self, test_set):
        e = 0
        classifier_error = 0
        test_set = test_set.to_numpy()
        for row in test_set:
            n = row.tolist()
            prob_vector = eval("self.theta"+str(n[:-1]))
            state = row[-1]  # get state 0-failed 1-success
            new_error = np.square(1-prob_vector[state])
            desicion = 1 if prob_vector[1] >= 0.5 else 0
            # calculate squared error
            e += new_error
            classifier_error += np.square(state - desicion)
        # get average error by dividing test size
        e /= len(test_set)
        classifier_error /= len(test_set)
        return e, classifier_error

    def chooseOrderMaximizingIG(self):
        indecies_of_data_rows = range(len(self.data))
        current_round_candidates = np.random.random_integers(
            0, len(self.data), 10)
