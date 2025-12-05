import numpy as np

def MSE_gradient(actual_outputs, values_vector , parameters_vector):
    # Implement the gradient calculation for Mean Squared Error
    n = len(actual_outputs)

    pred = np.dot(values_vector, parameters_vector)
    
    gradient_vector = 2/n * np.dot(values_vector.T, (pred - actual_outputs))
    return gradient_vector

class gradient_descent:

    def __init__(self, learning_rate=0.01, max_iterations=1000):
        
        self.name = "Gradient Descent Optimizer"

        # cost_function should be a callable that takes parameters and returns a scalar cost
        self.cost_function = None
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def step(self, actual_outputs, values_vector , parameters_vector):

        if self.cost_function is None:
            raise ValueError("Cost function not defined.")
        
        elif self.cost_function == 'MSE':
            self.gradient_vector = MSE_gradient(actual_outputs, values_vector , parameters_vector)
        
        next_step_parameters_vector = parameters_vector - self.learning_rate * self.gradient_vector
        
        return next_step_parameters_vector