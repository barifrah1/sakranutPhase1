from data_loader import DataLoader
from utils import args
from bayesClassifier import BayesClassifier

if __name__ == '__main__':
    data_loader = DataLoader(args)
    # preprocessing
    data, columnsInfo = data_loader.preprocess()
    # split data to train and test
    train, test_s = data_loader.split_train_test()
    bayes = BayesClassifier(train, columnsInfo)
    # print test error for internak use
    print('first test error: ', bayes.calculate_test_error(test_s)[
          0], ' classifing first test error: ', bayes.calculate_test_error(test_s)[1])
    new_theta = bayes.fit()
    print('final test error: ', bayes.calculate_test_error(test_s)[
          0], ' classifing final test error: ', bayes.calculate_test_error(test_s)[1])
    # plot IG graph
    bayes.plot_dkl_graph()
