import numpy as np
import pandas as pd 

data = pd.read_csv("kc_house_data.csv") # Import data

# Split training and testing dataset
div = int(len(data)*0.8)
train = data[:div]
test = data[div:]

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

def gradient_descent(features_matrix, output, parameters, step_size, tolerance):
    converged = False 
    
    # when gradient of RSS is larger than tolerance
    while not converged:
        # prediction = dot product between feature matrix and parameter
        # aka sum of (feature 1 x parameter 1), (feature 2 x parameter 2)... 
        pred = np.dot(features_matrix, parameters)
        error = output - pred # error vector
        grad_rss = 0 
        
        # loop through each data (house)
        for i in range(len(parameters)): 
            # compute partial derivative of RSS with respect to parameter
            partial_der = -2 * np.dot(error, features_matrix[:, i])
            # sum up the square of the partial derivative
            grad_rss += partial_der ** 2
            # update the parameter to be a better fit with the given stepsize
            parameters[i] = parameters[i] - (step_size * partial_der)
            # the magnitude of the RSS gradient = square root of sums of squared partial derivatives
            grad_magnitude = np.sqrt(grad_rss)
        
        # if the RSS gradient is smaller than tolerance, end loop
        if grad_magnitude < tolerance:
            converged = True            
    
    return parameters

def rss(output, features_matrix, parameters): # residual sums of squares
    y_hat = np.dot(features_matrix, parameters) # predicted output
    rss = np.sum(np.power((output - y_hat), 2))
    return rss

def r_2(output, rss): # coefficient of determination
    y_bar = np.mean(output) # mean of known outputs
    tss = np.sum(np.power((output - y_bar), 2)) # total sums of squares
    r2 = 1 - rss/tss
    return np.round(r2, 2)

# test the function for linear regression
feature1 = ['sqft_living']
output = 'price'
(feature_matrix1, output_array1) = get_np_data(train, feature1, output)
# initial parameters, stepsize, and tolerance are given by the 
initial_parameters1 = np.array([-49860., 1.])
step_size1 = 7e-12
tolerance1 = 2.5e7

parameters1 = np.round(gradient_descent(feature_matrix1, output_array1, initial_parameters1, 
                                             step_size1, tolerance1), 2)
rss1 = rss(output_array1, feature_matrix1, parameters1)
r2_1 = r_2(output_array1, rss1)
print("MODEL 1: x1 = house area (in sqft)")
print("MODEL 1 FORMULA: y = "+ str(parameters1[0]) + " + " + str(parameters1[1]) + "x1")
print("R^2 of MODEL 1:", r2_1)

# test the function for multiple regression (multiple inputs)
feature2 = ['sqft_living', 'sqft_living15']
(feature_matrix2, output_array2) = get_np_data(train, feature2, output)
initial_parameters2 = np.array([-100000., 5., 5.])
step_size2 = 4e-12
tolerance2 = 1e9

parameters2 = np.round(gradient_descent(feature_matrix2, output_array2, initial_parameters2, 
                                             step_size2, tolerance2), 2)
rss2 = rss(output_array2, feature_matrix2, parameters2)
r2_2 = r_2(output_array2, rss2)
print("\nMODEL 2: x1 = house area (in sqft), x2 = average house area of neighboring 15 houses")
print("MODEL 2: y = "+ str(parameters2[0]) + " + " + str(parameters2[1]) + "x1 + " 
      + str(parameters2[2]) + "x2")
print("R^2 of MODEL 2:", r2_2)
      





