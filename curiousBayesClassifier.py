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

    def calculate_step_without_update(self, row):
        n = row.tolist()
        state = row[-1]  # last element in vector  is the state
        # get the priors probabilities vector for current row
        p_before = np.array(eval("self.theta"+str(n[:-1])))
        #p_before = np.array(p_before_as_list)
        # Bayesian updates
        new_p = np.array([0, 0])
        sum_p = 0  # initialize sum using for normalization part of bayesian update
        for t in range(2):
            # p = theta[x]
            p = scipy.stats.norm(t, 0.5).pdf(state)
            # theta[t|x] ~ theta[t] * theta[x]
            # the next expression equals to  self.theta[n0, n1, n2, n3, n4,..., t] *= p
            exec("new_p[t]" +
                 "= self.theta"+str(n[:-1]+[t])+" * p")
            # this is for the normalization
            sum_p += new_p[t]
        # normalization - divide each element by norm
        for t in range(2):
            new_p[t] /= sum_p
        # posterior probabilities
        p_after = new_p.flatten()
        # compute IG between p_before and p_after
        D_KL_t = self.compute_kl_div(p_before, p_after)
        return D_KL_t, p_after

    def fit(self):
        iter = 1
        for _ in tqdm(range(len(self.data))):
            candidates = self.generateCurrentRoundCandidates(3)
            best_IG_cand_p = self.chooseCandMaximizingIG(candidates)
            chosenRow = self.data[best_IG_cand_p[0]]
            self.indecies_of_data_rows.remove(best_IG_cand_p[0])
            self.learning_order.append(best_IG_cand_p[0])
            exec("self.theta"+chosenRow[:-1] + "= best_IG_cand_p[2]")
            self.D_KL.append(best_IG_cand_p[1])
            if(iter % 10000 == 0):
                print('mean D_KL for iter ', iter, 'is: ', np.mean(self.D_KL))
            iter += 1
        return self.theta

    def plot_dkl_graph(self):
        plt.plot(self.D_KL)
        plt.show()

    def chooseCandMaximizingIG(self, candidates):
        best_IG = (-1, -1, np.array([0.5, 0.5]))
        for cand in candidates:
            ig, p_after = self.calculate_step_without_update(self.data[cand])
            if(ig > best_IG):
                best_IG = (cand, ig, p_after)
        return best_IG

    def generateCurrentRoundCandidates(self, numCandidates):
        current_round_candidates = np.random.choice(
            self.indecies_of_data_rows, numCandidates, replace=False)
        return current_round_candidates
