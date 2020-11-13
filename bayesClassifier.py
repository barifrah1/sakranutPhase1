import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from sakranot import DataLoader


class BayesClassifier:

    def __init__(self, data, columnsInfo):
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
        # unifor distribution over all variables
        self.theta = np.ones(numOfUniqueValues)/multiplication
        return self.theta

    def step(self, row):
        n0 = row[0]
        n1 = row[1]
        n2 = row[2]
        n3 = row[3]
        n4 = row[4]
        state = row[-1]
        p_before = np.array(self.theta[n0, n1, n2, n3, n4])

        # Bayesian updates
        sum_p = 0
        for t in range(2):
            # p = theta[x]
            p = scipy.stats.norm(t, 0.5).pdf(state)
            # theta[t|x] ~ theta[t] * theta[x]
            self.theta[n0, n1, n2, n3, n4, t] *= p
            # this is for the normalization
            sum_p += self.theta[n0, n1, n2, n3, n4, t]
        # normalization
        for t in range(2):
            self.theta[n0, n1, n2, n3, n4, t] /= sum_p
        p_after = self.theta[n0, n1, n2, n3, n4, :].flatten()
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
        for i in tqdm(range(len(self.data))):
            step_dkl = self.step(self.data[i])
            self.D_KL.append(step_dkl)
        return self.theta

    def plot_dkl_graph(self):
        plt.plot(self.D_KL)
        plt.show()

    def calculate_test_error(self, test_set):
        e = 0
        test_set = test_set.to_numpy()
        for i in range(1, len(test_set)):
            row = test_set[i]
            e += np.square(row[5]-self.theta[row[0], row[1],
                                             row[2], row[3], row[4], row[5]])
            e /= len(row)
        return e
