import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Data generation function to create a checkerboard-patterned dataset
def make_data_checkerboard(n, noise=0):
    n_samples = int(n / 4)
    scale = 5
    shift = 2.5
    center = 0.5
    c1a = (np.random.rand(n_samples, 2) - center) * scale + [-shift, shift]
    c1b = (np.random.rand(n_samples, 2) - center) * scale + [shift, -shift]
    c0a = (np.random.rand(n_samples, 2) - center) * scale + [shift, shift]
    c0b = (np.random.rand(n_samples, 2) - center) * scale + [-shift, -shift]
    X = np.concatenate((c1a, c1b, c0a, c0b), axis=0)
    y = np.concatenate((np.ones(2 * n_samples), np.zeros(2 * n_samples)))
    # Randomly flips a fraction of the labels to add noise
    for i, value in enumerate(y):
        if np.random.rand() < noise:
            y[i] = 1 - value
    return (X, y)


# Training datasets (we create 3 to use to average over model)
np.random.seed(88)
N = 3
X_train = []
y_train = []
for i in range(N):
    Xt, yt = make_data_checkerboard(500, noise=0.25)
    X_train.append(Xt)
    y_train.append(yt)

# Validation and test data
X_val, y_val = make_data_checkerboard(3000, noise=0.25)
X_test, y_test = make_data_checkerboard(3000, noise=0.25)

# For the final performance evaluation, train on all of the training and validation data:
X_train_plus_val = np.concatenate((X_train[0], X_train[1], X_train[2], X_val), axis=0)
y_train_plus_val = np.concatenate((y_train[0], y_train[1], y_train[2], y_val), axis=0)


################################
def_para = {
    "learning_rate_init" : 0.03,
    "hidden_layer_sizes" : (30,30),
    "alpha" : 0,
    "solver" : 'sgd',
    "tol" : 1e-5,
    "early_stopping" : False,
    "activation" : 'relu',
    "n_iter_no_change" : 1000,
    "batch_size" : 50,
    "max_iter" : 500
}


################################
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

def subplot_decision_boundary(model, X_train, y_train, sub_ax):
    step_size = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    model.fit(X_train, y_train)

    df = pd.DataFrame(data={"x0": [x[0] for x in X_train],
                            "x1": [x[1] for x in X_train]})

    # Plot the decision boundary.
    x_min, x_max = int(df.x0.min() - 2), int(df.x0.max() + 2)
    y_min, y_max = int(df.x1.min() - 2), int(df.x1.max() + 2)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.subplot(*sub_ax)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # Plot also the training points
    sns.scatterplot(data = df, x="x0", y="x1", hue=y_train, palette="deep")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()


#======================
# Single
'''
mpl = MLPClassifier(**def_para)
subplot_decision_boundary(mpl, X_train[0], y_train[0], (2,2,1))
'''

#======================

'''
para_hls = [(2, 2), (5, 5), (30, 30)]
para_lr = [0.0001, 0.01, 1]
para_reg = [0, 1, 10]
para_bz = [5, 50, 500]

change_para = {
    "hidden_layer_sizes": para_hls,
    "learning_rate_init": para_lr,
    "alpha": para_reg,  # regularization
    "batch_size": para_bz
}


index = 0
plt.figure(figsize = (16,16))

for para, vals in change_para.items():
    for v in vals:
        index += 1

        print(index)
        print("k : {} --- v: {}".format(para, v))

        arg = def_para.copy()
        arg[para] = v

        print(arg)

        #mpl = MLPClassifier(**arg)
        #subplot_decision_boundary(mpl, X_train[0], y_train[0], (1, 3, index))
'''

def evaluate_model_once(model, X_train, y_train):
    model.fit(X_train, y_train)
    accu = model.score(X_val, y_val)
    return accu


def evaluate_model(para_dic):
    accu_list = []
    for i in range (0, 3):
        mpl = MLPClassifier(**para_dic)
        accu_list.append(evaluate_model_once(mpl, X_train[i], y_train[i]))
    return np.average(accu_list)


def evaluate_model_with_diff_para(name, values, xlab, xlog=False):
    accu_hist = []
    for r in values:
        arg = def_para.copy()
        arg[name] = r
        accu_hist.append(evaluate_model(arg))
    ax = sns.lineplot(x=values, y=accu_hist)

    if xlog:
        ax.set(xscale="log")

    plt.title("Accuracy for different {}".format(xlab))

    if xlog:
        plt.xlabel("{} (Log)".format(xlab))
    else:
        plt.xlabel(xlab)
    plt.ylabel("Accuracy")
    plt.show()

lr = np.logspace(-5, 0, num=4)
evaluate_model_with_diff_para("learning_rate_init",
                              lr,
                              "Learning Rate",
                              xlog = True)




