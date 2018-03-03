from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import iris_sk as iris
import plot_decision_regions as pdr
import DT

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
forest.fit(iris.X_train, iris.y_train)
pdr.plot_decision_regions(DT.X_combined, DT.y_combined,
                            classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
