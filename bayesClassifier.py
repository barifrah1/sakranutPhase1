import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import itertools
import tqdm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


class BayesClassifier:
    # constructor
    def __init__(self, data, columnsInfo):
        if(isinstance(data, pd.DataFrame)):  # if data is dataframe - convert to array
            self.data = data.to_numpy(copy=True)
        else:
            self.data = data
        # columns of data and num of unique values - ('country',3) for example
        self.columnsInfo = columnsInfo
        print(self.columnsInfo)
        self.D_KL = []
        self.initialize_priors()  # define theta function

    def initialize_priors(self):
        # columnsInfo_structure - 'COLUMN_NAME': index 0, 'COLUMN_UNIQUE_VALUES': index 1
        numOfUniqueValues = np.fromiter(
            map(lambda x: x[1], self.columnsInfo), dtype=np.int)  # create a vector of num of unique value for each column
        # uniform distribution over all variables
        # 2 options: succuess of falil each has 0.5 probabilty at start time
        self.theta = np.ones(numOfUniqueValues)/2
        self.theta[:, 2, 1] = 0.69
        self.theta[:, 2, 0] = 0.31
        self.theta[:,1,1]=0.65
        self.theta[:,1,0]=0.35
        self.theta[:,0,1]=0.41
        self.theta[:,0,0]=0.59
        self.theta[:,0,:,0]=0.95
        self.theta[:,0,:,1]=0.05
        return self.theta

    def step(self, row):
        n = row.tolist()
        state = row[-1]  # last element in vector  is the state
        # get the priors probabilities vector for current row
        p_before = np.array(eval("self.theta"+str(n[:-1])))
        # p_before = np.array(p_before_as_list)
        # Bayesian updates
        sum_p = 0  # initialize sum using for normalization part of bayesian update
        for t in range(2):
            # p = theta[x]
            p = scipy.stats.norm(t, 0.5).pdf(state)
            # theta[t|x] ~ theta[t] * theta[x]
            # the next expression equals to  self.theta[n0, n1, n2, n3, n4,..., t] *= p
            exec("self.theta"+str(n[:-1]+[t]) +
                 "= self.theta"+str(n[:-1]+[t])+" * p")
            # this is for the normalization
            sum_p += eval("self.theta"+str(n[:-1]+[t]))
        # normalization - divide each element by norm
        for t in range(2):
            exec("self.theta"+str(n[:-1]+[t]) +
                 "/= sum_p ")
        # posterior probabilities
        p_after = eval("self.theta"+str(n[:-1])+".flatten()")
        # compute IG between p_before and p_after
        D_KL_t = self.compute_kl_div(p_before, p_after)
        return D_KL_t
    # get priors and posterior of specific row and compute IG

    def compute_kl_div(self, p_before, p_after):
        D_KL_t = 0.0
        for t in range(2):
            if p_before[t] > 0.0 and p_after[t] > 0.0:
                D_KL_t += p_after[t] * np.log2(p_after[t] / p_before[t])
        return D_KL_t
    #

    def fit(self):
        iter = 1
        for row in tqdm(self.data):
            step_dkl = self.step(row)
            self.D_KL.append(step_dkl)
            if(iter % 10000 == 0):
                print('mean D_KL for iter ', iter, 'is: ', np.mean(self.D_KL))
            iter += 1
        return self.theta
    # plot IG graph of all iterations

    def plot_dkl_graph(self):
        compressed_IG = []
        for iter in range(int(len(self.D_KL)/1000)):
            compressed_IG.append(
                np.mean(self.D_KL[iter*1000:(iter*1000+1000)]))
        plt.plot(compressed_IG)
        plt.show()

    # calculate test error - for inside purposes only
    def calculate_test_error(self, test_set):
        e = 0
        pred = []
        classifier_error = 0
        test_set = test_set.to_numpy()
        for row in test_set:
            n = row.tolist()
            prob_vector = eval("self.theta"+str(n[:-1]))
            state = row[-1]  # get state 0-failed 1-success
            new_error = np.square(1-prob_vector[state])
            desicion = 1 if prob_vector[1] >= 0.5 else 0
            pred.append(desicion)
            # calculate squared error
            e += new_error
            classifier_error += np.square(state - desicion)
        # get average error by dividing test size
        e /= len(test_set)
        classifier_error /= len(test_set)
        return e, classifier_error

    def confusion_matrix_and_auc(self, test_set):
        pred = []
        test_set = test_set.to_numpy()
        for row in test_set:
            n = row.tolist()
            prob_vector = eval("self.theta"+str(n[:-1]))
            state = row[-1]  # get state 0-failed 1-success
            desicion = 1 if prob_vector[1] >= 0.5 else 0
            pred.append(desicion)
        cm = confusion_matrix(
            test_set[:, -1], pred, np.array([1, 0]), normalize='true')
        auc = roc_auc_score(test_set[:, -1], pred)
        return (cm, auc)
