'''
    There are in total five algorithms: DT, SVM, SGD, and GB

'''

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

global seed
seed = 0

class Ensemble_Classifier():
    def __init__(self):
        self.SVM = SVC(kernel = 'linear')
        self.GB = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        self.DT = DecisionTreeClassifier(random_state=seed)
        self.SGD = SGDClassifier(random_state=seed)
        # initialize the ratio of total classifiers
        self.ratio = 0.25 * np.ones((4,))

    # define the fit function for ensemble classifier
    def fit(self, X, y):
        # chose the same test_size with GAN model
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5)
        self.DT.fit(xtrain, ytrain)
        self.SGD.fit(xtrain, ytrain)
        self.GB.fit(xtrain, ytrain)
        self.SVM.fit(xtrain, ytrain)
        # compute the accuracy of classifiers on testing data
        DT_score = self.DT.score(xtest, ytest)
        SGD_score = self.SGD.score(xtest, ytest)
        GB_score = self.GB.score(xtest, ytest)
        SVM_score = self.SVM.score(xtest, ytest)
        self.alpha = np.array([DT_score, SGD_score, GB_score, SVM_score])
        # calculate the percent of different score values
        self.alpha = self.alpha/np.sum(self.alpha)

    # define the predict function for ensemble classifier
    def predict(self, X):
        DT_predict = self.DT.predict(X)
        SVM_predict = self.SVM.predict(X)
        GB_predict = self.GB.predict(X)
        SGD_predict = self.SGD.predict(X)
        Ensemble_predict = self.alpha[0] * DT_predict + self.alpha[1] * SGD_predict + \
                       self.alpha[2] * GB_predict + self.alpha[3] * SVM_predict
        # filter the low score
        Ensemble_predict = np.ones(Ensemble_predict.shape) * (Ensemble_predict > 0.5)
        return Ensemble_predict

    # define the score function for ensemble classifier
    def score(self, X, y):
        return accuracy_score(y, self.predict(X), sample_weight=None)
