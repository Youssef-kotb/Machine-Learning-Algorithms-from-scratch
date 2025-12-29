import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KNN:

    def __init__(self, k_neighbors=3):
        self.k_neighbors = k_neighbors

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, Input):

        list_of_predictions = [self._predict_single(x) for x in Input]

        return np.array(list_of_predictions)
    
    
    def _predict_single(self, single_Input):

        distances = [euclidean_distance(single_Input, x) for x in self.x_train]

        # getting the nearst K neighbors
        k_indeces = np.argsort(distances)[:self.k_neighbors] # returns the sorted INDECES

        k_lables = [y for y in self.y_train[k_indeces]]

        # getting the most common class label
        most_common = Counter(k_lables).most_common(1)

        return most_common[0][0]