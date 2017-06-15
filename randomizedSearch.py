from sklearn.grid_search import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
Y = load_iris().target
k_range = range(1,31)
weight_options=['uniform','distance']
param_grad = dict(n_neighbors=k_range, weights=weight_options)

kenighbor = KNeighborsClassifier(n_neighbors=5)
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(kenighbor, param_grad, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, Y)
    best_scores.append(round(rand.best_score_, 3))
print(str(best_scores))