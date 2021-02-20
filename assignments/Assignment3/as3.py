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
        if proba > thresh:
            return 1
        else:
            return 0

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

import seaborn as sns
import matplotlib.pyplot as plt

LR = [10e-2, 10e-4, 10e-6, 1]
np.random.seed(1234)

init_w = random_sample(3)

for lr in LR:
    reg = Logistic_regression()
    reg.fit(X_train, Y_train, init_w, lr=lr, max_iter=5000)
    cost = reg.learning_curve(X_train, Y_train)
    plt.plot(cost, label='learning rate={}'.format(lr))

# Output picture
plt.xlabel('Number of iteration')
plt.ylabel('Cost')
plt.title('Cost function under different learning rates over 50 iterations')
plt.legend()
plt.show()





















