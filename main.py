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
    # split data to train and test
    train, test_s = data_loader.split_train_test()

    # for comparison:
    clf = RandomForestClassifier(
        n_estimators=500, random_state=0)
    clf.fit(train.to_numpy()[:, :-1], train.to_numpy()[:, -1])
    p = clf.predict(test_s.to_numpy()[:, :-1])
    auc = roc_auc_score(test_s.to_numpy()[:, -1], p)
    print("rf auc:", auc)
    bayes = BayesClassifier(train, columnsInfo)
    # print test error for internal use
    print('first test error: ', bayes.calculate_test_error(test_s)[
          0], ' classifing first test error: ', bayes.calculate_test_error(test_s)[1])
    print('cm:', bayes.confusion_matrix_and_auc(test_s)[
        0], 'auc: ', bayes.confusion_matrix_and_auc(test_s)[1])
    new_theta = bayes.fit()
    print('final test error: ', bayes.calculate_test_error(test_s)[
          0], ' classifing final test error: ', bayes.calculate_test_error(test_s)[1])
    print('cm:', bayes.confusion_matrix_and_auc(test_s)[
          0], 'auc: ', bayes.confusion_matrix_and_auc(test_s)[1])
    # plot IG graph
    bayes.plot_dkl_graph()
    smartBayes = CuriousBayesClassifier(train, columnsInfo)
    # print test error for internal use
    print('first test error: ', smartBayes.calculate_test_error(test_s)[
          0], ' classifing first test error: ', smartBayes.calculate_test_error(test_s)[1])
    print('cm:', smartBayes.confusion_matrix_and_auc(test_s)[
        0], 'auc: ', smartBayes.confusion_matrix_and_auc(test_s)[1])
    new_theta = smartBayes.fit(args['numCandidatesInIter'])
    print('final test error: ', smartBayes.calculate_test_error(test_s)[
          0], ' classifing final test error: ', smartBayes.calculate_test_error(test_s)[1])
    print('cm:', smartBayes.confusion_matrix_and_auc(test_s)[
        0], 'auc: ', smartBayes.confusion_matrix_and_auc(test_s)[1])
    # plot IG graph
    smartBayes.plot_dkl_graph()
