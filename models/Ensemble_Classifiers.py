'''
    There are in total five algorithms: MLP, SVM, Bag, AdaBoost and GB

'''

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

global seed
seed = 0

class Ensemble_Classifier():
    def __init__(self):
        self.MLP = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                                                  solver='sgd', verbose=0, tol=1e-4, random_state=seed,
                                                  learning_rate_init=.1)
        self.SVM = SVC(kernel = 'linear')
        self.Bag = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=seed)
        self.AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=seed)
        self.GB = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        self.ratio = 0.2 * np.ones((5,))

    # define the fit function for ensemble classifier
    def fit(self, X, y):
        # chose the same test_size with GAN model
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        self.MLP.fit(xtrain, ytrain)
        self.Bag.fit(xtrain, ytrain)
        self.AdaBoost.fit(xtrain, ytrain)
        self.GB.fit(xtrain, ytrain)
        self.SVM.fit(xtrain, ytrain)
        # compute the accuracy of classifiers on testing data
        MLP_score = self.MLP.score(xtest, ytest)
        Bag_score = self.Bag.score(xtest, ytest)
        AdaBoost_score = self.AdaBoost.score(xtest, ytest)
        GB_score = self.GB.score(xtest, ytest)
        SVM_score = self.SVM.score(xtest, ytest)
        self.alpha = np.array([MLP_score, Bag_score, AdaBoost_score, GB_score, SVM_score])
        # calculate the percent of different score values
        self.alpha = self.alpha/np.sum(self.alpha)

    # define the predict function for ensemble classifier
    def predict(self, X):
        MLP_predict = self.MLP.predict(X)
        SVM_predict = self.SVM.predict(X)
        AdaBoost_predict = self.AdaBoost.predict(X)
        GB_predict = self.GB.predict(X)
        Bag_predict = self.Bag.predict(X)
        Ensemble_predict = self.alpha[0] * MLP_predict + self.alpha[1] * SVM_predict + self.alpha[2] * AdaBoost_predict + \
                       self.alpha[3] * GB_predict + self.alpha[4] * Bag_predict
        # filter the low score
        Ensemble_predict = np.ones(Ensemble_predict.shape) * (Ensemble_predict > 0.5)
        return Ensemble_predict

    # define the score function for ensemble classifier
    def score(self, X, y):
        return accuracy_score(y, self.predict(X), sample_weight=None)
