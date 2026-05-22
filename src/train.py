import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import clean_data
from src.features import create_features
from src.utils import calculate_distance

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Load data
df = load_data("data/train.csv")

# Clean
df = clean_data(df)

# Features
df = create_features(df)

# Distance
df["distance_km"] = df.apply(lambda row: calculate_distance(
    row["pickup_latitude"],
    row["pickup_longitude"],
    row["dropoff_latitude"],
    row["dropoff_longitude"]
), axis=1)


# Features & Target
X = df[["passenger_count", "hour", "day", "month", "distance_km"]]
y = df["fare_amount"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")