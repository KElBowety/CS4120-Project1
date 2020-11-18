import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('SGD:', np.mean(y_pred == y_test))

ab = AdaBoostClassifier()

ab.fit(X_train, y_train)

y_pred = ab.predict(X_test)

print('AdaBoost:', np.mean(y_pred == y_test))

gp = GaussianProcessClassifier()

gp.fit(X_train, y_train)

y_pred = gp.predict(X_test)

print('Gaussian Process:', np.mean(y_pred == y_test))

