from data_loader import DataLoader
from utils import args
from bayesClassifier import BayesClassifier
from curiousBayesClassifier import CuriousBayesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    data_loader = DataLoader(args)
    # preprocessing
    data, columnsInfo = data_loader.preprocess()
    # regular bayes classifier
    bayes = BayesClassifier(data, columnsInfo)
    new_theta = bayes.fit()
    # plot IG graph
    bayes.plot_dkl_graph()
    # curious bayes classifier
    smartBayes = CuriousBayesClassifier(data, columnsInfo)
    new_theta = smartBayes.fit(args['numCandidatesInIter'])
    # plot IG graph
    smartBayes.plot_dkl_graph()
