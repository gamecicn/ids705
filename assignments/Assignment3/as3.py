import numpy as np
import pandas as pd






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
        # returns the average cross entropy cost
        size = X.shape[0]
        z = np.dot(X, w.T)
        return np.sum(np.log(1 + np.exp(z)) - np.dot(y,z)) / size

    def gradient(self, w, X, y):
        size = X.shape[0]
        X = self.prepare_x(X)
        return np.dot(X.T, self.sigmoid(X, self.w) - y) / size

    def norm(self, w):
        return np.linalg.norm(w, ord=2)

    # Update the weights in an iteration of gradient descent
    def gradient_descent(self, X, y, lr):
        # returns s scalar of the magnitude of the Euclidean norm
        #  of the change in the weights during one gradient descent step
        pre_norm = self.norm(self.w)
        self.w = self.w - lr * self.gradient(self.w, X, y)
        cur_norm = self.norm(self.w)
        return abs(pre_norm - cur_norm)


    # Fit the logistic regression model to the data through gradient descent
    def fit(self, X, y, w_init, lr, delta_thresh=1e-6, max_iter=5000, verbose=False):
        # Note the verbose flag enables you to print out the weights at each iteration
        #  (optional - but may help with one of the questions)
        self.w = w_init
        step_size = 1
        step = 0

        while True:
            # Check stop conditions
            if step_size < delta_thresh:
                break

            step += 1
            if step > max_iter:
                break

            step_size = self.gradient_descent(X, y, lr)


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
        pass

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
LR = [10e-2, 10e-4, 10e-6, 10e0]


np.random.seed(1234)
cost = []
w = np.random.random_sample(3)

lreg = Logistic_regression()

for i in LR:
    w, cost_train, cost_test = lreg.gradient_descent(X_train, Y_train, X_test, Y_test, i)


























