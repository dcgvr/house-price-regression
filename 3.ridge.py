import numpy as np
import pandas as pd

# Import data
train = pd.read_csv("kc_house_train_data.csv")
test = pd.read_csv("kc_house_test_data.csv")

def get_np_data(data, features, output):
    # first parameter is y-intercept so should be constant (multiply by 1)
    features_matrix = np.ones((len(data), 1)) 
    
    # locate wanted features (e.g. house area) and attach to the feature matrix
    # convert Pandas Dataframe to Numpy Array
    for f in features:
        feature_pd = data.loc[:, f]
        feature_np = feature_pd.to_numpy()
        features_matrix = np.column_stack((features_matrix, feature_np))
    
    # extract output feature (price)
    output_pd = data.loc[:, output]
    output_array = output_pd.to_numpy()
    
    return features_matrix, output_array

def gradient_descent(features_matrix, output, parameters, step_size, l2, max_iteration):
    converged = False 
    count = 0
    # when gradient of RSS is larger than tolerance
    while not converged:
        # prediction = dot product between feature matrix and parameter
        # aka sum of (feature 1 x parameter 1), (feature 2 x parameter 2)... 
        pred = np.dot(features_matrix, parameters)
        error = output - pred # error vector
        
        # loop through each data (house)
        for i in range(len(parameters)): 
            # compute partial derivative of the cost function with respect to parameter 
            # first half (-2 * error * feature)) represents the RSS
            # second half (2 * l2 * i) represents the regularization
            # optimize RSS = avoid underfitting, regularization = avoid overfitting
            if i == 0:
                partial_der = -2 * np.dot(error, features_matrix[:, i])
            else:
                partial_der = -2 * np.dot(error, features_matrix[:, i]) + 2 * l2 * parameters[i]
            # update the parameter to be a better fit with the given stepsize
            parameters[i] = parameters[i] - (step_size * partial_der)
            
        # add 1 to the counter
        count += 1
        
        # if the RSS gradient is smaller than tolerance, end loop
        if count == max_iteration:
            converged = True            
    
    return parameters

def rss(output, features_matrix, parameters): # residual sums of squares
    y_hat = np.dot(features_matrix, parameters) # predicted output
    rss = np.sum(np.power((output - y_hat), 2))
    return rss

# Define initial conditions for the regression and extract data
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_np_data(train, model_features, my_output)
(test_feature_matrix, test_output) = get_np_data(test, model_features, my_output)

# Least-squared Regression (minimize RSS, minimize bias)
para_0_penalty = np.round(gradient_descent(feature_matrix, output, [0, 0, 0], 1e-12, 0, 3000), 2)
rss_0_penalty = rss(test_output, test_feature_matrix, para_0_penalty)
print()
print("\nMODEL 1: x1 = house area (in sqft), x2 = average house area of neighboring 15 houses")
print("MODEL 1: y = "+ str(para_0_penalty[0]) + " + " + str(para_0_penalty[1]) + "x1 + " 
      + str(para_0_penalty[2]) + "x2")
print("RSS of MODEL 1:", rss_0_penalty)

# Ridge Regression (balance bias and variance to avoid overfitting)
para_high_penalty = np.round(gradient_descent(feature_matrix, output, [0, 0, 0], 1e-12, 1e11, 3000), 2)
rss_high_penalty = rss(test_output, test_feature_matrix, para_high_penalty)
print("\nMODEL 2: x1 = house area (in sqft), x2 = average house area of neighboring 15 houses")
print("MODEL 2: y = "+ str(para_high_penalty[0]) + " + " + str(para_high_penalty[1]) + "x1 + " 
      + str(para_high_penalty[2]) + "x2")
print("RSS of MODEL 2:", rss_high_penalty)


