import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import sklearn.gaussian_process.kernels as kernels
import csv


def hcvDataset():
    hcv = pd.read_csv("HCV-Egy-Data.csv")

    X = hcv.iloc[:, :27]
    X = np.asarray(X)
    y = hcv.iloc[:, 28]
    y = np.asarray(y)

    # Age
    X[:, 0] = np.digitize(X[:, 0], [31, 37, 42, 47, 52, 57, 62], right=True)
    # BMI
    X[:, 2] = np.digitize(X[:, 2], [21, 25, 30, 35, 40], right=False)
    # WBC
    X[:, 10] = np.digitize(X[:, 10], [4000, 11000, 12102], right=False)
    # RBC
    X[:, 11] = np.digitize(X[:, 11], [3000000, 5000000, 5018451], right=False)
    # Plat (ASK ABOUT DISCRETIZATION MISTAKE)
    X[:, 13] = np.digitize(X[:, 13], [100000, 226465, 255000], right=False)
    # AST1
    X[:, 14] = np.digitize(X[:, 14], [20, 40.5, 129], right=False)
    # ALT1
    X[:, 15] = np.digitize(X[:, 15], [20, 40.5, 129], right=False)
    # ALT4
    X[:, 16] = np.digitize(X[:, 16], [20, 40.5, 129], right=False)
    # ALT12
    X[:, 17] = np.digitize(X[:, 17], [20, 40.5, 129], right=False)
    # ALT24
    X[:, 18] = np.digitize(X[:, 18], [20, 40.5, 129], right=False)
    # ALT36
    X[:, 19] = np.digitize(X[:, 19], [20, 40.5, 129], right=False)
    # ALT48
    X[:, 20] = np.digitize(X[:, 20], [20, 40.5, 129], right=False)
    # ALT after 24
    X[:, 21] = np.digitize(X[:, 21], [20, 40.5, 129], right=False)
    # RNA Base
    X[:, 22] = np.digitize(X[:, 22], [5, 1201086], right=True)
    # RNA 4
    X[:, 23] = np.digitize(X[:, 23], [5, 1201715], right=True)
    # RNA 12
    X[:, 24] = np.digitize(X[:, 24], [5, 3731527], right=True)
    # RNA EOT
    X[:, 25] = np.digitize(X[:, 25], [5, 808450], right=True)
    # RNA EF
    X[:, 26] = np.digitize(X[:, 26], [5, 808450], right=True)

    for i in range(len(X)):
        # HGB
        if X[i][1] == 1:
            if X[i][12] < 14:
                X[i][12] = 0
            elif X[i][12] <= 17.5:
                X[i][12] = 1
            elif X[i][12] <= 20:
                X[i][12] = 2
        else:
            if X[i][12] < 12.3:
                X[i][12] = 0
            elif X[i][12] <= 15.3:
                X[i][12] = 1
            elif X[i][12] <= 20:
                X[i][12] = 2


def sgd(xdata, ydata):
    loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    penalty = ['l2', 'l1', 'elasticnet']
    alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    max_iter = [100, 500, 1000, 5000, 10000, 50000]
    learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
    early_stopping = [False, True]
    warm_start = [False, True]

    with open('sgd.csv', mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['loss', 'penalty', 'alpha', 'max_iter', 'learning_rate',
                         'early_stopping', 'warm_start', 'accuracy'])
        for l in loss:
            for p in penalty:
                for a in alpha:
                    for m in max_iter:
                        for lr in learning_rate:
                            for e in early_stopping:
                                for w in warm_start:
                                    accuracy = 0
                                    model = SGDClassifier(loss=l, penalty=p, alpha=a, max_iter=m, learning_rate=lr,
                                                          early_stopping=e, eta0=0.01, warm_start=w, random_state=1)
                                    kf = StratifiedKFold(n_splits=5, shuffle=True)
                                    for i, j in kf.split(xdata, ydata):
                                        X_ktrain, X_ktest = X[i], X[j]
                                        y_ktrain, y_ktest = y[i], y[j]
                                        model.fit(X_ktrain, y_ktrain)
                                        ypred = model.predict(X_ktest)
                                        accuracy += np.mean(ypred == y_ktest)
                                    accuracy /= 5
                                    writer.writerow([l, p, a, m, lr, e, w, accuracy])


def adaboost(xdata, ydata):
    base_estimator = [DecisionTreeClassifier(max_depth=1),
                      DecisionTreeClassifier(max_depth=5),
                      DecisionTreeClassifier(max_depth=10),
                      DecisionTreeClassifier(max_depth=50),
                      DecisionTreeClassifier(max_depth=100)
                      ]
    n_estimators = [5, 10, 25, 50, 75, 100]
    learning_rate = [0.1, 0.5, 1.0, 2.0]
    algorithm = ['SAMME', 'SAMME.R']

    with open('adaboost.csv', mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'accuracy'])
        for b in base_estimator:
            for n in n_estimators:
                for l in learning_rate:
                    for a in algorithm:
                        accuracy = 0
                        model = AdaBoostClassifier(base_estimator=b, n_estimators=n, learning_rate=l,
                                                   algorithm=a, random_state=1)
                        kf = StratifiedKFold(n_splits=5, shuffle=True)
                        for i, j in kf.split(xdata, ydata):
                            X_ktrain, X_ktest = X[i], X[j]
                            y_ktrain, y_ktest = y[i], y[j]
                            model.fit(X_ktrain, y_ktrain)
                            ypred = model.predict(X_ktest)
                            accuracy += np.mean(ypred == y_ktest)
                        accuracy /= 5
                        writer.writerow([b, n, l, a, accuracy])


def gp(xdata, ydata):
    kernel = [kernels.RBF(),
              kernels.Matern(),
              kernels.ConstantKernel(),
              kernels.WhiteKernel(),
              kernels.RationalQuadratic()
              ]
    max_iter_predict = [10, 50, 100, 500, 1000]
    warm_start = [False, True]
    multi_class = ['one_vs_rest', 'one_vs_one']

    with open('gaussianprocess.csv', mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['kernel', 'max_iter_predict', 'warm_start', 'multi_class', 'accuracy'])
        for k in kernel:
            for m in max_iter_predict:
                for w in warm_start:
                    for mc in multi_class:
                        accuracy = 0
                        model = GaussianProcessClassifier(kernel=k, max_iter_predict=m, warm_start=w,
                                                          multi_class=mc, random_state=1)
                        kf = StratifiedKFold(n_splits=5, shuffle=True)
                        for i, j in kf.split(xdata, ydata):
                            X_ktrain, X_ktest = X[i], X[j]
                            y_ktrain, y_ktest = y[i], y[j]
                            model.fit(X_ktrain, y_ktrain)
                            ypred = model.predict(X_ktest)
                            accuracy += np.mean(ypred == y_ktest)
                        accuracy /= 5
                        writer.writerow([k, m, w, mc, accuracy])


if __name__ == '__main__':
    bn = pd.read_csv('haberman.csv')

    X = bn.iloc[:, :3]
    X = np.asarray(X)
    y = bn.iloc[:, 3]
    y = np.asarray(y)

    # norm = Normalizer()
    # norm.fit(X)
    # X = norm.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    gp(X_train, y_train)

    # ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), random_state=1)
    #
    # ab.fit(X_train, y_train)
    #
    # y_pred = ab.predict(X_test)
    #
    # print('AdaBoost:', np.mean(y_pred == y_test))
    #
    # gp = GaussianProcessClassifier(kernel=kernels.WhiteKernel(), random_state=1)
    #
    # gp.fit(X_train, y_train)
    #
    # y_pred = gp.predict(X_test)
    #
    # print('Gaussian Process:', np.mean(y_pred == y_test))
