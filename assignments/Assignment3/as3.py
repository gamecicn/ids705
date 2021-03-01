# Load the MNIST Data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set this to True to download the data for the first time and False after the first time 
#   so that you just load the data locally instead
download_data = False

if download_data:
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

    # Adjust the labels to be '1' if y==3, and '0' otherwise
    y[y != '3'] = 0
    y[y == '3'] = 1
    y = y.astype('int')

    # Divide the data intro a training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=88)

    file = open('./tmpdata', 'wb')
    pickle.dump((X_train, X_test, y_train, y_test), file)
    file.close()
else:
    file = open('./tmpdata', 'rb')
    X_train, X_test, y_train, y_test = pickle.load(file)
    file.close()



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

lr_min = LogisticRegression(penalty = "l1", C = pow(10, 100) , solver='liblinear')
lr_best = LogisticRegression(penalty = "l1", C = pow(10, -2) , solver='liblinear')
lda = LDA()
rf = RandomForestClassifier()

pred_list = []
prob_list = []

for m in [lda, rf, lr_min, lr_best]:
    m.fit(X_train, y_train)
    pred_list.append(m.predict(X_test))
    prob_list.append(m.predict_proba(X_test)[:, 1])

print("done")