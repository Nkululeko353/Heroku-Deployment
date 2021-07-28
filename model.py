# Importing the libraries

import pandas as pd
import pickle

data = pd.read_csv('Advertising.csv')
data

data.shape
data.describe()


# What are the features?

# TV: advertising dollars spent on TV for a single product in a given market 
#    (in thousands of dollars)
# Radio: advertising dollars spent on Radio
# Newspaper: advertising dollars spent on Newspaper

# What is the response?

# Sales: sales of a single product in a given market (in thousands of items)

# What else do we know?

# Because the response variable is continuous, this is a regression problem.
# There are 200 observations (represented by the rows), and each observation is a single market.


# create a Python list of feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the DataFrame
y = data['sales']

# We have to drop the variable Unnamed:0
data.drop(['Unnamed: 0'], axis=1)


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('lr_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('lr_model.pkl','rb'))
print(model.predict([[2, 9, 6]]))