# Import necessary libraries
import pandas as pd

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into an 80-20 training-test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Print the scaled training and test sets
print("Scaled training features: \n" , X_train_scaled)
print("Scaled test features: \n" , X_test_scaled)
