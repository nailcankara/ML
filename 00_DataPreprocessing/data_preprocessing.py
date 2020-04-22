import numpy as np  #mathematical computation
import matplotlib.pyplot as plt   #graph
import pandas as pd   #dataset operations

dataset = pd.read_csv("Data.csv")     #read csv
X = dataset.iloc[: , :-1].values      #independent variables
y = dataset.iloc[: , 3:].values      #dependent variables

from sklearn.preprocessing import Imputer  #Convert NaN values to numerical values
imputer = Imputer(missing_values="NaN" , strategy="mean" , axis=0)   
imputer = imputer.fit(X[:, 1:3])   #fit selected columns
X[:,1:3] = imputer.transform(X[:,1:3]) #tranform selected columns

from sklearn.preprocessing import LabelEncoder #Convert Categorial values to numerical values
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[: , 0])  #fit and transform selected columns
y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder      #for dummy variables
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split   #Split data
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2 , random_state = 0)

from sklearn.preprocessing import StandardScaler   #Standardization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
