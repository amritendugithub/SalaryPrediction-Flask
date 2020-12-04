import numpy as np 
import pandas as pd 
import pickle

# Data Source : https://www.kaggle.com/rsadiq/salary

df = pd.read_csv("Salary.csv")
## This displays the top 5 rows of the data
# print(df.head())

## Provides some information regarding the columns in the data
#print(df.info())

## This describes the basic stat behind the dataset used 
#print(data.describe())

X = df.iloc[:,:-1]
y= df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error
# calculate Mean square error
mse = mean_squared_error(y_test,y_pred)
# Calculate R square vale
rsq = r2_score(y_test,y_pred)
print('mean squared error :',mse)
print('r square :',rsq)

# Create a pickle file using serialization of the Classifier Model
import pickle
pickle_out = open("lr.pkl","wb")
pickle.dump(lr,pickle_out)
pickle_out.close()