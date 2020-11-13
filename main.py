from sakranot import DataLoader
from utils import args
from bayesClassifier import BayesClassifier

if __name__ == '__main__':
    data_loader = DataLoader(args)
    data, columnsInfo = data_loader.preprocess()
    train, test_s = data_loader.split_train_test()
    bayes = BayesClassifier(train, columnsInfo)
    print('first test error: ', bayes.calculate_test_error(test_s))
    new_theta = bayes.fit()
    print('final test error: ', bayes.calculate_test_error(test_s))
    bayes.plot_dkl_graph()
