# implementing Logistic Regression from scratch
import numpy as np

def run_gradient_descent(X, y, weights, epochs, optimizer):

    print("Performing Gradient Descent for", epochs, "epochs.")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        weights = optimizer.step(actual_outputs=y, values_vector=X, parameters_vector=weights)

    return weights

class LogisticRegression:

    # the constructor, it initializes the optimizer and cost function
    def __init__(self, optimizer, cost_function='binary_crossentropy', threshold=0.5):
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.optimizer.cost_function = cost_function
        self.threshold = threshold

    # method to fit the model to the training data
    def fit(self, X, y, epochs=100):

        print("Fitting the model using", self.optimizer, "with", self.cost_function, "cost function.")

        self.x_train = X
        self.y_train = y

        # adding the bias term to X in the first column
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias

        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        if self.optimizer.name == "Gradient Descent Optimizer":
            final_weights = run_gradient_descent(X, y, self.weights, epochs, self.optimizer)

        self.weights = final_weights
        return self.weights
    
    def predict(self, X):
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias

        linear_model = np.dot(X, self.weights)
        predictions = 1 / (1 + np.exp(-linear_model))  # Sigmoid function
        predictions = (predictions > self.threshold).astype(int)

        return predictions
    
    def predict_proba(self, X):
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias

        linear_model = np.dot(X, self.weights)
        probabilities = 1 / (1 + np.exp(-linear_model))  # Sigmoid function

        return probabilities