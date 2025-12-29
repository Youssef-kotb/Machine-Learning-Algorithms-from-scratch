import numpy as np


class MSE:
    def __init__(self):
        pass

    def forward(self, parameters_vector, X_values_vector, true_outputs):
        """
        Compute the Mean Squared Error loss.

        Parameters:
        parameters_vector (list or array-like): Model weights.
        X_values_vector (list or array-like): True data points.
        true_outputs (list or array-like): True outputs.

        Returns:
        float: The Mean Squared Error loss.

        """
        y_pred = np.dot(X_values_vector, parameters_vector)
 
        n = len(y_pred)
        mse = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, true_outputs)) / n

        return mse
    
    def backward(self, parameters_vector, X_values_vector, true_outputs):
        """
        Compute the gradient of the Mean Squared Error loss with respect to predictions.

        Parameters:
        parameters_vector (list or array-like): Model weights.
        X_values_vector (list or array-like): True data points.
        true_outputs (list or array-like): True outputs.

        Returns:
        list: Gradient of the loss with respect to each predicted value.
        """
        
        y_pred = np.dot(X_values_vector, parameters_vector)
        n = len(y_pred)
        gradient = (2/n) * np.dot(X_values_vector.T, (y_pred - true_outputs))
        return gradient