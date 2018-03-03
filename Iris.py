#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import perceptron as perc
import plot_decision_regions as pdr
import adalinegd as ada
import adalinesgd as sgd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50,0], X[:50,1],
	color='red', marker='o', label='setosa')
plt.scatter(X[:50,0], X[50:100,1],
	color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

ppn = perc.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

pdr.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 =ada.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
    np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(SSE)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = ada.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
    np.log10(ada1.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('SSE')
ax[1].set_title('Adaline - Learning rate 0.0001')

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada3 = ada.AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
pdr.plot_decision_regions(X_std, y, classifier=ada3)
plt.title('Adaline _ Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()


ada4 = sgd.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada4.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada4)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal legnth [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada4.cost_) + 1), ada4.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()