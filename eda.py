import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/train.csv")


# =====================================================
# BASIC DATA UNDERSTANDING
# =====================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE VALUES")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())


# =====================================================
# DATA CLEANING
# =====================================================

# Remove negative fare amounts
df = df[df["fare_amount"] > 0]

# Keep valid passenger counts
df = df[
    (df["passenger_count"] > 0)
    & (df["passenger_count"] <= 6)
]

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nCLEANED DATA SHAPE")
print(df.shape)


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Convert pickup_datetime to datetime format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Extract useful time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month


# =====================================================
# DISTANCE CALCULATION FUNCTION
# =====================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371  # Earth radius in kilometers

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


# =====================================================
# CREATE DISTANCE FEATURE
# =====================================================

df["distance_km"] = df.apply(
    lambda row: calculate_distance(
        row["pickup_latitude"],
        row["pickup_longitude"],
        row["dropoff_latitude"],
        row["dropoff_longitude"]
    ),
    axis=1
)

print("\nDISTANCE FEATURE")
print(df[["distance_km", "fare_amount"]].head())


# =====================================================
# MACHINE LEARNING
# =====================================================

# Select input features
X = df[
    [
        "passenger_count",
        "hour",
        "day",
        "month",
        "distance_km"
    ]
]

# Target variable
y = df["fare_amount"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# MODEL TRAINING
# =====================================================

model = LinearRegression()

model.fit(X_train, y_train)


# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# MODEL EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)


# =====================================================
# FINAL DATASET PREVIEW
# =====================================================

print("\nFINAL DATASET")
print(df.head())

# Save trained model

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")