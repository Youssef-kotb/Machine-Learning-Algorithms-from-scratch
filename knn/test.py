from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import knn

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knnModel = knn.knn(k_neighbors=3)
knnModel.fit(X_train, y_train)

predictions = knnModel.predict(X_test)

print("Predictions:", predictions)
print("True Labels:", y_test)