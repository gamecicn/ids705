import time
import numpy as np
import pandas as pd
import scipy.stats as ss

def euclidean_distances(pl, p):
    return np.reshape(np.sqrt(np.sum((pl - p) ** 2, axis=1)), (-1, 1))

#def euclidean_distances_multi_to_multi(pl1, pl2):
#return np.apply_along_axis(euclidean_distances_multi_to_one, pl2, pl1)



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

        for r in np.array(x):
            dist = euclidean_distances(np.array(self.x), r)
            vec = list(map(lambda p: 1 if p[1] < k else 0, enumerate(ss.rankdata(dist) - 1)))
            val = np.dot(vec, self.y)
            y_hat.append(1 if (val > (k / 2)) else 0)

        return y_hat


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


K = 5
knn = Knn()

### Low dimension
time_begin = time.time()
knn.fit(X_train_low, y_train_low)
y_hat    = knn.predict(X_test_low, K)
accu = accuracy(y_test_low.iloc[:,0], y_hat)
duration = time.time() - time_begin

print("[Low dimension] Accuracy [{:0.4f}] \t Time [{:0.4f}] Sec".format(accu, duration))