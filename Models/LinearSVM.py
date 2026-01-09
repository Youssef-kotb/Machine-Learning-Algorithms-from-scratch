import numpy as np
from Losses.HingeLoss import HingeLoss

hinge = HingeLoss()
   

class SVMLoss:
    def __init__(self, C):
        self.C = C
        
    def forward(self, parameters_vector, X_values_vector, true_outputs):
        
        w = parameters_vector
        
        scores = X_values_vector @ w

        hinge_loss = hinge.forward(true_outputs, scores)

        return 0.5 * np.dot(w, w) + self.C * np.mean(hinge_loss)
    
    def backward(self, parameters_vector, X_values_vector, true_outputs):

        w = parameters_vector
        
        scores = X_values_vector @ w

        grad_scores = hinge.backward(true_outputs,scores)

        
        gradients = w + self.C * (X_values_vector.T @ grad_scores) / X_values_vector.shape[0]

        return gradients
    

class LinearSVM:
    def __init__(self, optimizer, C = 100):
        self.optimizer = optimizer
        self.optimizer.cost_function = SVMLoss(C)


    def fit(self, X, y, epochs = 100):

        print("Fitting the model using", self.optimizer, "with SVMLoss cost function. (built in)")
        
        self.x_train = X
        self.y_train = y

        self.y_train = change_range(self.y_train)
        
        #debug print
        print(self.y_train)

        #adding the bias term to X in the first column
        x_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X = x_bias

        #initialize weights
        self.weights = np.zeros(X.shape[1])


        # Run optimization
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.weights = self.optimizer.step(true_outputs=y, X_values_vector=X , parameters_vector=self.weights)
            print("Weights:", self.weights)
        return self.weights
    
    def predict(self, X):

        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        boundary = X_bias @ self.weights

        return np.sign(boundary)



def change_range(y):
        
    values = np.unique(y)

    if len(values) > 2 :
        raise Exception("Linear SVM is a binary classifier, only two classes can be classified for now, multi SVM is comming soon, stay tuned")
    
    y = y.copy()
    y[y == values[0]] = -1
    y[y == values[1]] = 1

    return y


