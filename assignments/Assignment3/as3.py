import numpy as np
import pandas as pd
from numpy.random import random_sample




# Logistic regression class
class Logistic_regression:
    # Class constructor
    def __init__(self):
        self.w = None  # logistic regression weights
        self.saved_w = []  # Since this is a small problem, we can save the weights
        #  at each iteration of gradient descent to build our
        #  learning curves
        # returns nothing
        pass

    # Method for calculating the sigmoid function of w^T X for an input set of weights
    def sigmoid(self, X, w):
        # returns the value of the sigmoid
        return 1 / (1 + np.exp(-np.dot(X, w.T)))

    # Cost function for an input set of weights
    def cost(self, X, y, w):
        N = X.shape[0]
        X = self.prepare_x(X)
        sig = self.sigmoid(X, w)
        return (np.dot(-y, np.log(sig)) - np.dot((1 - y).T, np.log(1 - sig))) / N

    def gradient(self, w, X, y):
        size = X.shape[0]
        return np.dot(X.T, self.sigmoid(X, w) - y) / size

    def norm(self, w):
        return np.linalg.norm(w, ord=2)

    # Update the weights in an iteration of gradient descent
    def gradient_descent(self, X, y, lr):
        # returns s scalar of the magnitude of the Euclidean norm
        #  of the change in the weights during one gradient descent step
        self.w = self.w - lr * self.gradient(self.w, X, y)
        return self.norm(self.saved_w[-1] - self.w)

    def __refesh_weight(self, w_init):
        self.saved_w.clear()
        self.saved_w.append(self.w)

    # Fit the logistic regression model to the data through gradient descent
    def fit(self, X, y, w_init, lr, delta_thresh=1e-6, max_iter=5000, verbose=False):
        # Note the verbose flag enables you to print out the weights at each iteration
        #  (optional - but may help with one of the questions)
        self.w = w_init
        step_size = 1
        step = 0

        X = self.prepare_x(X)

        self.__refesh_weight(w_init)

        while True:
            # Check stop conditions
            if step_size < delta_thresh:
                break

            step += 1
            if step > max_iter:
                break

            # One step gradient descent
            step_size = self.gradient_descent(X, y, lr)

            # Save weight
            self.saved_w.append(self.w)

    # Use the trained model to predict the confidence scores (prob of positive class in this case)
    def predict_proba(self, X):
        # returns the confidence score for the each sample
        if self is None:
            exit("The weights is empty")
        X = self.prepare_x(X)
        return self.sigmoid(X, self.w)

    # Use the trained model to make binary predictions
    def predict(self, X, thresh=0.5):
        # returns a binary prediction for each sample
        proba = self.predict_proba(X)
        return np.array(list(map(int, np.array(proba > 0.5))))


    # Stores the learning curves from saved weights from gradient descent
    def learning_curve(self, X, y):
        # returns the value of the cost function from each step in gradient descent
        #  from the last model fitting process
        cost = []
        for w in self.saved_w:
            cost.append(self.cost(X, y, w))
        return cost

    # Appends a column of ones as the first feature to account for the bias term
    def prepare_x(self, X):
        # returns the X with a new feature of all ones (a column that is the new column 0)
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)



TRAIN_PERCENT = 0.7

df = pd.read_csv("./data/A3_Q1_data.csv")

train_size = int(df.shape[0] * TRAIN_PERCENT)


X_train = df.iloc[:train_size, :][["x1", "x2"]]
X_test  = df.iloc[train_size:, :][["x1", "x2"]]
Y_train = df.iloc[:train_size, :]["y"]
Y_test  = df.iloc[train_size:, :]["y"]

print(X_train.shape)


##############################################################
# h
##############################################################

from sklearn.neighbors import KNeighborsClassifier

init_w = random_sample(3)


import numpy as np
from sklearn import metrics
from numpy.random import random_sample

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

### Test model performance through cross validation
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True)
kf_lr_pred = []
kf_knn_pred = []
kf_answer = []

for train_index, val_index in kf.split(X_train, Y_train):

    # Split data
    cv_train_X, cv_valid_X = X_train.iloc[train_index], X_train.iloc[val_index]
    cv_train_y, cv_valid_y = Y_train[train_index], Y_train[val_index]

    # Training
    reg = Logistic_regression()
    reg.fit(cv_train_X, cv_train_y, init_w, lr=1, max_iter=500)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)

    # Predict
    kf_lr_pred.extend(reg.predict(cv_valid_X))
    kf_knn_pred.extend(knn.predict(cv_valid_X))
    kf_answer.extend(cv_valid_y)


fpr, tpr, thres = metrics.roc_curve(kf_answer, kf_lr_pred, pos_label=1)
auc = metrics.roc_auc_score(kf_answer, kf_lr_pred)
legend_string = 'AUC = {:0.3f}'.format(auc)






