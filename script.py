import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers.head())

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].apply(lambda x:'1' if x=='female' else '0')
print(passengers['Sex'].head())

# Fill the nan values in the age column
passengers['Age'].fillna(value=round(passengers['Age'].mean()), inplace=True)
print(passengers['Age'].values)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x:1 if x==1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x:1 if x==2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age','FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split

X_train, X_test, y_train, y_test = train_test_split(features, survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

#Create and train the model
mlr = LogisticRegression()
mlr.fit(X_train, y_train)

#score the model on train data
print(mlr.score(X_train, y_train))

#score the model on test data
print(mlr.score(X_test, y_test))

#Anlyze the coefficient
print(mlr.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,18.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
print(scaler.transform(sample_passengers))

# Make survival predictions!
print(mlr.predict(sample_passengers))
print(mlr.predict_proba(sample_passengers))