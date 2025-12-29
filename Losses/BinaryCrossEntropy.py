import numpy as np

class BinaryCrossEntropy:
    
    def __init__(self):
        pass

    def forward(self, parameters_vector, X_values_vector, true_outputs):
        """
        Compute the Binary Cross-Entropy loss.

        Parameters:
        parameters_vector (list or array-like): Model weights.
        X_values_vector (list or array-like): True data points.
        true_outputs (list or array-like): True outputs.

        Returns:
        float: The Binary Cross-Entropy loss.

        """
        y_pred = 1 / (1 + np.exp(-np.dot(X_values_vector, parameters_vector)))  # Sigmoid function

        n = len(y_pred)
        bce = -sum(yt * np.log(yp + 1e-15) + (1 - yt) * np.log(1 - yp + 1e-15) for yp, yt in zip(y_pred, true_outputs)) / n

        return bce
    
    def backward(self, parameters_vector, X_values_vector, true_outputs):
        """
        Compute the gradient of the Binary Cross-Entropy loss with respect to predictions.

        Parameters:
        parameters_vector (list or array-like): Model weights.
        X_values_vector (list or array-like): True data points.
        true_outputs (list or array-like): True outputs.

        Returns:
        list: Gradient of the loss with respect to each predicted value.
        """
        y_pred = 1 / (1 + np.exp(-np.dot(X_values_vector, parameters_vector)))  # Sigmoid function
        n = len(y_pred)
        gradient = 1/n * np.dot(X_values_vector.T, (y_pred - true_outputs))
        return gradient