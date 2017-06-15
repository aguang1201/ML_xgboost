import xgboost as xgb
import numpy as np
import graphviz
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
Y = load_iris().target
k_range = range(1,31)
weight_options=['uniform','distance']
param_grad = dict(n_neighbors=k_range, weights=weight_options)

kenighbor = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(kenighbor,param_grid=param_grad,cv=10,scoring='accuracy')
grid.fit(X, Y)

grid.grid_scores_
print(str(grid.best_score_))
print(str(grid.best_params_))

# data = np.random.rand(5,10)
# label = np.random.randint(2,size=5)
# dtrain = xgb.DMatrix(data=data,label=label)
# dtrain.save_binary('train.buffer')
# weight = np.random.rand(5,1)
# dtrain = xgb.DMatrix(data=data, label=label, missing= -999.0, weight=weight)
# param =