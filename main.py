import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/train.csv")

print("Dataset loaded:", df.shape)

# Feature engineering (same as app.py logic)
df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
df["day"] = pd.to_datetime(df["pickup_datetime"]).dt.day
df["month"] = pd.to_datetime(df["pickup_datetime"]).dt.month

df["distance_km"] = (
    (df["pickup_longitude"] - df["dropoff_longitude"])**2 +
    (df["pickup_latitude"] - df["dropoff_latitude"])**2
) ** 0.5 * 111

# Remove nulls
df = df.dropna()

# Features & Target
X = df[[
    "passenger_count",
    "hour",
    "day",
    "month",
    "distance_km"
]]

y = df["fare_amount"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 RANDOM FOREST MODEL (UPGRADED)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Random Forest Model trained & saved successfully")