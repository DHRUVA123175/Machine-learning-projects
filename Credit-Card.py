
#CREDIT CARD FRAUDULENT USING DEEP LEARNING(TENSORFLOW)

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from  sklearn.metrics import classification_report,confusion_matrix

file=r"C:\Users\ABCD\Downloads\creditcard.csv\creditcard.csv"
data=pd.read_csv(file)
data.head()


# Separate fraud and non-fraud cases
fraud = data[data["Class"] == 1]
non_fraud = data[data["Class"] == 0]

# Oversample the fraud cases to match the number of non-fraud cases
fraud_upsampled = resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42)

# Create a balanced dataset
balanced_set = pd.concat([non_fraud, fraud_upsampled])

# Separate features and target variable
X = balanced_set.drop("Class", axis=1)
y = balanced_set["Class"]

# Normalize only the feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Ensure y_train and y_test have correct shape
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

print("Data processing completed successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Define Model (Fixed Input Layer & Loss Function)
model = Sequential([
    Dense(units=50, activation="relu", input_shape=(X_train.shape[1],)),  # Direct input shape
    Dropout(0.4),
    Dense(units=25, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

# Compile Model (Fixed Loss Function Name)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Ensure correct shape
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Ensure data type compatibility
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Check shapes before training
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt
from  sklearn.metrics import classification_report,confusion_matrix

plt.plot(history.history["accuracy"],label=["Train accuracy"])
plt.plot(history.history["val_accuracy"],label=["Validation accuracy"])
plt.legend()
plt.show()

predictions=(model.predict(X_test)>0.5).astype("int32")

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

sample_transaction=X_test[0:1]
prediction=model.predict(sample_transaction)

print("Fradualant" if prediction > 0.5 else "Genuine transaction")

#DOWNLOAD THE CSV FILE WRT CREDIT-CARD FROM KAGGLE WEBSITE(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?)