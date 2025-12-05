from gradient_descent.gradient_descent import gradient_descent
import numpy as np
from linear_regression.linear_regression import linear_regression

# fake data for testing
values_vector = np.array([[2],
                          [3],
                          [4],
                          [5]])

actual_outputs = np.array([7,
                           6,
                           5,
                           4])

opt = gradient_descent(learning_rate=0.05)

model = linear_regression(optimizer=opt, cost_function='MSE')

model.fit(values_vector, actual_outputs, epochs=1000)

#new data for prediction
X_new = np.array([[6],
                  [7],
                  [8]])

# predu=ictions should be close to [3, 2, 1]

y_new = np.array([3, 2, 1])
predictions = model.predict(X_new)

print("Predictions for new data:", predictions)
