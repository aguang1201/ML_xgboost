from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
Y = load_iris().target

knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X,Y)
knn.predict([3,5,4,2])