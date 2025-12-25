# implementing Naive Bayes classifier from scratch
import numpy as np

class naive_bayes:
    def __init__(self, alpha = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        num_classes = len(np.unique(y))
        num_features = X.shape[1]
        num_samples = X.shape[0]

        # calculate prior probabilities
        self.class_priors = np.zeros(num_classes)

        for c in range(num_classes):
            self.class_priors[c] = np.sum(y == c) / num_samples

        # calculate likelihoods with Laplace smoothing
        self.likelihoods = np.zeros((num_classes, num_features))

        for c in range(num_classes):
            X_c = X[y == c] # get all samples belonging to class c

            
            self.likelihoods[c, :] = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + self.alpha * 2)
        
    
    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.class_priors)
        log_priors = np.log(self.class_priors)
        log_likelihoods = np.log(self.likelihoods)
        log_likelihoods_neg = np.log(1 - self.likelihoods)

        y_pred = np.zeros(num_samples)

        for i in range(num_samples):
            log_posteriors = np.zeros(num_classes)

            for c in range(num_classes):
                log_posteriors[c] = log_priors[c] + np.sum(X[i, :] * log_likelihoods[c, :] + (1 - X[i, :]) * log_likelihoods_neg[c, :])

            y_pred[i] = np.argmax(log_posteriors)

        return y_pred