from sklearn.svm import SVC
import iris_sk as iris
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(iris.X_train_std, iris.y_train)
pdr.plot_decision_regions(iris.X_combined_std, iris.y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# SVC(kernel='rbf', gamma=0.10, C=10.0, random_state=0)
