import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


class BayesClassifier:

    def __init__(self, data, columnsInfo):
        if(isinstance(data, pd.DataFrame)):
            self.data = data.to_numpy(copy=True)
        else:
            self.data = data
        self.columnsInfo = columnsInfo
        self.initialize_priors()

    def initialize_priors(self):
        # columnsInfo_structure - 'COLUMN_NAME': index 0, 'COLUMN_UNIQUE_VALUES': index 1
        numOfUniqueValues = np.fromiter(
            map(lambda x: x[1], self.columnsInfo), dtype=np.int)
        # start with uniform distribution over all paramters
        multiplication = 1
        for x in numOfUniqueValues:
            multiplication *= x
        """self.theta = {}
        for col in self.columnsInfo:
            S[col[0]] = range(col[1])
        for comb in iterate_values(S):"""

        # uniform distribution over all variables
        self.theta = np.ones(numOfUniqueValues)/2
        return self.theta

    def step(self, row):
        n = []
        for i in range(len(row)):
            n.append(row[i])
        """n0 = row[0]
        n1 = row[1]
        n2 = row[2]
        n3 = row[3]
        n4 = row[4]"""
        state = row[-1]
        # normalization
        #norm = sum(self.theta[n0, n1, n2, n3, n4])
        """if(norm != 1):
            p_before = np.array(self.theta[n0, n1, n2, n3, n4])/norm
        else:"""
        p_before_as_list = eval("self.theta"+str(n[:-1]))
        p_before = np.array(p_before_as_list)
        # Bayesian updates
        sum_p = 0
        for t in range(2):
            # p = theta[x]
            p = scipy.stats.norm(t, 0.5).pdf(state)
            # theta[t|x] ~ theta[t] * theta[x]
            # self.theta[n0, n1, n2, n3, n4, t] *= p
            exec("self.theta"+str(n[:-1]+[t]) +
                 "= self.theta"+str(n[:-1]+[t])+" * p")
            # this is for the normalization
            sum_p += eval("self.theta"+str(n[:-1]+[t]))
        # normalization
        for t in range(2):
            exec("self.theta"+str(n[:-1]+[t]) +
                 "/= sum_p ")
        p_after = eval("self.theta"+str(n[:-1])+".flatten()")
        D_KL_t = self.compute_kl_div(p_before, p_after)
        return D_KL_t

    def compute_kl_div(self, p_before, p_after):
        D_KL_t = 0.0
        for t in range(2):
            if p_before[t] > 0.0 and p_after[t] > 0.0:
                D_KL_t += p_after[t] * np.log2(p_after[t] / p_before[t])
        return D_KL_t

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

    def plot_dkl_graph(self):
        plt.plot(self.D_KL)
        plt.show()

    def calculate_test_error(self, test_set):
        e = 0
        test_set = test_set.to_numpy()
        for row in test_set:
            n = []
            for i in range(len(row)):
                n.append(row[i])
            prob_vector = eval("self.theta"+str(n[:-1]))
            state = row[5]
            new_error = np.square(1-prob_vector[state])
            e += new_error
        e /= len(test_set)
        return e


def iterate_values(S):
    keys, values = zip(*S.items())
    for row in itertools.product(*values):
        yield dict(zip(keys, row))
