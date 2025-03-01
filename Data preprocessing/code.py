# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Feature matrix (X) and dependent variable vector (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original Feature Matrix (X):\n", X)
print("\nOriginal Dependent Variable Vector (y):\n", y)

# Handling missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("\nFeature Matrix (X) after handling missing values:\n", X)

# Encoding the independent variable (One-Hot Encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("\nFeature Matrix (X) after One-Hot Encoding:\n", X)

# Encoding the dependent variable (Label Encoding)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
y = lb.fit_transform(y)

print("\nDependent Variable Vector (y) after Label Encoding:\n", y)

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nTraining Features (X_train):\n", X_train)
print("\nTesting Features (X_test):\n", X_test)
print("\nTraining Labels (y_train):\n", y_train)
print("\nTesting Labels (y_test):\n", y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("\nScaled Training Features (X_train):\n", X_train)
print("\nScaled Testing Features (X_test):\n", X_test)
