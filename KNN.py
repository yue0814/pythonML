from sklearn.neighbors import KNeighborsClassifier
import iris_sk as iris
import matplotlib.pyplot as plt
import plot_decision_regions as pdr

knn = KNeighborsClassifier(n_neighbors=5, p=2,
                            metric='minkowski')
knn.fit(iris.X_train_std, iris.y_train)
pdr.plot_decision_regions(iris.X_combined_std, iris.y_combined,
                            classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()
