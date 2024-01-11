import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Generate some random data for training purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a lnear regression model
model = LinearRegression()

#Train the model on the training set
model.fit(X_train, y_train)

#Make Predictions on test set
y_pred = model.predict(X_test)

#Evaluate the model perfomance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#Visualize the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression Model')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.show()
