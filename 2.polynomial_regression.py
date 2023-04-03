from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv") # Import data

# compute residual sums of squares & coefficient of determination
def rss_r2(y, y_hat):
    rss = np.sum(np.power((y - y_hat), 2))
    y_bar = np.mean(y) # mean of real y
    tss = np.sum(np.power((y - y_bar), 2)) # total sums of squares
    r2 = 1 - rss/tss
    
    return rss, r2

def poly_fit(train, test, degree):
    
    # convert Pandas DataFrames to Numpy Arrays
    x_train = train["sqft_living"].to_numpy()
    y_train = train["price"].to_numpy()
    x_test = test["sqft_living"].to_numpy()
    y_test = test["price"].to_numpy()
    
    # fit the data with a polynomial regression of a given degree
    # using scikit learning module
    poly_features= PolynomialFeatures(degree = degree)
    x_poly_train = poly_features.fit_transform(x_train.reshape(-1, 1))
    x_poly_test = poly_features.fit_transform(x_test.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly_train, y_train)
    y_hat_test = model.predict(x_poly_test) # predicted y
    
    rss_test, r2_test = rss_r2(y_test, y_hat_test)
    
    return rss_test, r2_test
                     
# Split data into training, validation and testing datasets (70% - 15% - 15%)
train = data.sample(frac = 0.7, random_state = 8)
non_train = data.drop(train.index)
validate = non_train.sample(frac = 0.15, random_state = 8)
test = non_train.drop(validate.index)

r2_validate_array = []
rss_validate_array = []

# Fit polynomial regression for degree 1 to 15
# Find R^2 and RSS of Validation Dataset for each degree fit
for i in range(1,16):
    rss_validate, r2_validate = poly_fit(train, validate, i)
    r2_validate_array.append(r2_validate)
    rss_validate_array.append(rss_validate)

# Plot R^2 and RSS as a function of polynomial regression degree
degree_array = np.arange(1, 16, 1)

plt.plot(degree_array, r2_validate_array, "ro", ms = 2)
plt.xlabel("Polynomial Regression Degree")
plt.ylabel("Model R^2")
plt.title("R^2 OF VALIDATION DATASET")
plt.show()

plt.plot(degree_array, rss_validate_array, "ro", ms = 2)
plt.xlabel("Polynomial Regression Degree")
plt.ylabel("Model RSS")
plt.title("RSS OF VALIDATION DATASET")
plt.show()

# Find optimized degree (maximum R^2)
best_degree = np.argmax(r2_validate_array) + 1
print("Optimized Polynomial Regression Degree:", best_degree)

# Find R^2 of Linear Regression and Optimized Polynomial Regression for Validation and Testing dataset
rss_validate1, r2_validate1 = poly_fit(train, validate, 1)
print("R^2 of Validation Dataset with Linear Regression:", np.round(r2_validate1, 2))
rss_validate2, r2_validate2 = poly_fit(train, validate, best_degree)
print("R^2 of Validation Dataset with Optimized Polynomial Regression:", np.round(r2_validate2, 2))

rss_test1, r2_test1 = poly_fit(train, test, 1)
print("R^2 of Testing Dataset with Linear Regression:", np.round(r2_test1, 2))
rss_test2, r2_test2 = poly_fit(train, test, best_degree)
print("R^2 of Testing Dataset with Optimized Polynomial Regression:", np.round(r2_test2, 2))


