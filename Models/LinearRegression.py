import numpy as np


class LinearRegression:
    def __init__(self, optimizer, cost_function):
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.optimizer.cost_function = cost_function

    def fit(self, X, y, epochs=100):


        print("Fitting the model using", self.optimizer, "with", self.cost_function, "cost function.")


        self.x_train = X
        self.y_train = y

        #adding the bias term to X in the first column
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias



        # Initialize weights
        self.weights = np.zeros(X.shape[1])



        # Run optimization
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.weights = self.optimizer.step(true_outputs=y, X_values_vector=X , parameters_vector=self.weights)

        return self.weights
    

    def predict(self, X):
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias
        

        predictions = np.dot(X, self.weights)
        return predictions
