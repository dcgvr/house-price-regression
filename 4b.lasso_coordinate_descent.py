import numpy as np
import pandas as pd
from math import sqrt

train = pd.read_csv("kc_house_train_data.csv")
test = pd.read_csv("kc_house_test_data.csv")

# decrease the weighted distance between big house and small house
train['sqft_living_sqrt'] = train['sqft_living'].apply(sqrt)
train['sqft_lot_sqrt'] = train['sqft_lot'].apply(sqrt)

# increase the weighted distance between many and not many floors/ bedrooms
train['bedrooms_square'] = train['bedrooms']*train['bedrooms']
train['floors_square'] = train['floors']*train['floors']

test['sqft_living_sqrt'] = test['sqft_living'].apply(sqrt)
test['sqft_lot_sqrt'] = test['sqft_lot'].apply(sqrt)
test['bedrooms_square'] = test['bedrooms']*test['bedrooms']
test['floors_square'] = test['floors']*test['floors']

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

# normalize the data so all the values are within similar ranges, which help with the regression
def normalize(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis = 0)
    feature_norm = feature_matrix / norms
    return norms, feature_norm

# each step of the coordinate descent algorithm
# LASSO COST: RSS + L1 * absolute_value(weights aka coefficients)
def lasso_cd_step(i, feature_matrix, output, weight, l1):
    predict = np.dot(feature_matrix, weight) # predicted values
    # partial derivative of the Lasso Cost with respect to the ith weight
    # all the maths are in this function are according to the course
    ro_i = np.sum(feature_matrix[:, i] * (output - predict + weight[i] * feature_matrix[:, i]))
    
    # adjust the weight accordingly to the partial derivative
    # i don't find the maths here intuitive but i grasp the big picture
    if i == 0: # the first weight - the intercept should remain unchanged
        new_i = ro_i
    # if the weight is too negative, it's increased to balance bias/variance
    elif ro_i < (-l1/2):
        new_i = ro_i + (l1/2)
    # if the weight is too positive, it's decreased to balance bias/variance
    elif ro_i > (l1/2):
        new_i = ro_i - (l1/2)
    # if the weight is insignificant in magnitude, we set the weight to 0
    # aka we don't consider that feature in our regression model
    else: 
        new_i = 0
        
    return new_i

# the full coordinate descent algorithm
def lasso_cd(feature_matrix, output, weight, l1, tolerance):
    converged = False
    n_col = np.size(feature_matrix, axis = 1) # number of features
    
    while not converged:
        change_list = []
        
        # adjust each weight using lasso_cd_step above
        for i in range(n_col):
            new_i = lasso_cd_step(i, feature_matrix, output, weight, l1)
            change_list.append(np.abs(weight[i] - new_i))
            weight[i] = new_i
        
        # if the maximum change in weight is lowered than tolerance
        # the weights are optimized for best regression model possible
        # loop is terminated
        if np.max(change_list) < tolerance:
            converged = True
            
    return weight

def rss(output, features_matrix, weights): # residual sums of squares
    y_hat = np.dot(features_matrix, weights) # predicted output
    rss = np.sum(np.power((output - y_hat), 2))
    return rss

def r_2(output, rss): # coefficient of determination
    y_bar = np.mean(output) # mean of known outputs
    tss = np.sum(np.power((output - y_bar), 2)) # total sums of squares
    r2 = 1 - rss/tss
    return np.round(r2, 2)

features = ['bedrooms', 'bedrooms_square','bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above','sqft_basement',
                'yr_built', 'yr_renovated']

# build a Lasso Regression Model based on all these features with normalized data
(feature_matrix, output) = get_np_data(train, features, 'price')
(norms, feature_matrix) = normalize(feature_matrix)

# obtain weights and normalize weights
initial_weights = np.zeros(len(features) + 1)
weight = lasso_cd(feature_matrix, output, initial_weights, 1e4, 5e5)
weight_norm = weight / norms
print("Lasso Model - multiply each entry of the array (coefficient) to the corresponding input feature (e.g. bedrooms): ", weight_norm)

# use normalize weights to predict output of test dataset
(test_feature_matrix, test_output) = get_np_data(test, features, 'price')
rss_test = rss(test_output, test_feature_matrix, weight_norm)
r2_test = r_2(test_output, rss_test)
print("R^2 of Lasso Model: ", r2_test)





        
        
        
        
        
        
        
        
        
