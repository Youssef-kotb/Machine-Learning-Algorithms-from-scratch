import numpy as np

class GradientDescent:

    def __init__(self, learning_rate=0.01):
        
        # cost_function should be a callable that takes parameters and returns a scalar cost
        self.cost_function = None
        self.learning_rate = learning_rate

    def step(self, true_outputs, X_values_vector , parameters_vector):

        if self.cost_function is None:
            raise ValueError("Cost function not defined.")
        
        else:
            self.gradient_vector = self.cost_function.backward(parameters_vector, X_values_vector, true_outputs)


        next_step_parameters_vector = parameters_vector - self.learning_rate * self.gradient_vector
        
        return next_step_parameters_vector