import time
import pandas as pd
import numpy as np
import scipy.stats as ss

from sklearn.metrics.pairwise import euclidean_distances


# (a) Write your own kNN classifier
class Knn:
    # k-Nearest Neighbor class object for classification training and testing
    def __init__(self):
        self.x = []
        self.y = []

    def fit(self, x, y):
        # Save the training data to properties of this class
        self.x = x
        self.y = y

    def predict(self, x, k):
        # Calculate the distance from each vector in x to the training data
        y_hat = []
        dist_matrix = euclidean_distances(self.x, x)

        for i, v in enumerate(x.values):
            distance = np.sum((self.x - v) ** 2, axis=1)
            distance_sorted = np.sort(distance)
            indices = np.where(distance <= distance_sorted[k - 1])
            vote = sum(self.y.iloc[indices])
            y_hat.append(vote > k / 2)

        return np.array(y_hat)

        '''
        top_points = np.apply_along_axis(lambda z: list(map(lambda p: 1 if p[1] < k else 0,
                                                            enumerate(ss.rankdata(z) - 1))),
                                         1, dist_matrix)

        return [1 if x else 0 for x in (np.dot(top_points, y_train_high) > k / 2)]
        '''

        '''
        for i, v in enumerate(x):
            diff = self.x - v

            distance = np.sum(diff ** 2, axis=1)
            distance_sorted = np.sort(distance)
            indices = np.where(distance <= distance_sorted[k - 1])
            vote = sum(self.y[indices])
            y_hat.append(vote > k / 2)

        # Return the estimated targets
        return np.array(y_hat)
        '''

# Metric of overall classification accuracy
#  (a more general function, sklearn.metrics.accuracy_score, is also available)
def accuracy(y, y_hat):
    nvalues = len(y)
    accuracy = sum(y == y_hat) / nvalues
    return accuracy


X_train_high = pd.read_csv("./data/A2_X_train_high.csv", header = None)
X_train_low = pd.read_csv("./data/A2_X_train_low.csv", header = None)
X_test_high = pd.read_csv("./data/A2_X_test_high.csv", header = None)
X_test_low = pd.read_csv("./data/A2_X_test_low.csv", header = None)

y_train_high = pd.read_csv("./data/A2_y_train_high.csv", header = None)
y_train_low = pd.read_csv("./data/A2_y_train_low.csv", header = None)
y_test_high = pd.read_csv("./data/A2_y_test_high.csv", header = None)
y_test_low = pd.read_csv("./data/A2_y_test_low.csv", header = None)




## C
K = 5
knn = Knn()

### Low dimension
time_begin = time.time()
knn.fit(X_train_low, y_train_low)
y_hat    = knn.predict(X_test_low, K)
accu = accuracy(y_test_low.iloc[:,0], y_hat)
duration = time.time() - time_begin

print("[Low dimension] Accuracy [{:0.4f}] \t Time [{:0.4f}]".format(accu, duration))

### High dimension
time_begin = time.time()
knn.fit(X_train_high, y_train_high)
y_hat    = knn.predict(X_test_high, K)
accu = accuracy(y_test_high.iloc[:,0], y_hat)
duration = time.time() - time_begin

print("[High dimension] Accuracy [{:0.4f}] \t Time [{:0.4f}]".format(accu, duration))













