import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model  # using scikit-learn

# data type list given by the course
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# import the whole dataset
sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)

# decrease the weighted distance between big house and small house
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)

# increase the weighted distance between many and not many floors/ bedrooms
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

# list of all input features 
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

# create a lasso regression model with L1 value = 500 and normalized data
model_all = linear_model.Lasso(alpha = 5e2, normalize = True) 
model_all.fit(sales[all_features], sales['price']) # return weights of the regression model
print(model_all.score(sales[all_features], sales['price']))

# import sub-datasets
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
valid = pd.read_csv('kc_house_valid_data.csv', dtype=dtype_dict)

# transform certain data the same way as explained 
test['sqft_living_sqrt'] = test['sqft_living'].apply(sqrt)
test['sqft_lot_sqrt'] = test['sqft_lot'].apply(sqrt)
test['bedrooms_square'] = test['bedrooms']*test['bedrooms']
test['floors_square'] = test['floors']*test['floors']

train['sqft_living_sqrt'] = train['sqft_living'].apply(sqrt)
train['sqft_lot_sqrt'] = train['sqft_lot'].apply(sqrt)
train['bedrooms_square'] = train['bedrooms']*train['bedrooms']
train['floors_square'] = train['floors']*train['floors']

valid['sqft_living_sqrt'] = valid['sqft_living'].apply(sqrt)
valid['sqft_lot_sqrt'] = valid['sqft_lot'].apply(sqrt)
valid['bedrooms_square'] = valid['bedrooms']*valid['bedrooms']
valid['floors_square'] = valid['floors']*valid['floors']

l1_penalty = np.logspace(1, 7, num = 13) # an array of tested L1 [10, 10^1.5, 10^2, ..., 10^7]
r2_train_list = []

# try out different L1 value to find the optimized one for the train dataset
for l1 in l1_penalty:
    model_train = linear_model.Lasso(alpha = l1, normalize = True)
    model_train.fit(train[all_features], train['price'])
    r2_train = model_train.score(train[all_features], train['price'])
    r2_train_list.append(r2_train)
print(r2_train_list)

best_l1 = l1_penalty[np.argmax(r2_train_list)]
print("Best L1:", best_l1)

# build a Lasso regression model with the optimized L1
model_train = linear_model.Lasso(alpha = best_l1, normalize = True)
model_train.fit(train[all_features], train['price'])

# apply the Lasso regression with optimized L1 to test dataset
r2_test = model_train.score(test[all_features], test['price'])
print("R^2 of Testing Dataset using Lasso Regression with Best L1 value:", r2_test)

n_select = np.count_nonzero(model_train.coef_) + np.count_nonzero(model_train.intercept_)
print("Number of selected features:", n_select)
    
    
    
    
    
    
    