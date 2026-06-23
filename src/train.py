import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/telco_churn_cleaned.csv")

# Drop customer ID
df = df.drop("customerID", axis=1)

# Encode target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encoding
df = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Best model
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Save model
joblib.dump(
    model,
    "models/churn_model.pkl"
)

print("Model saved successfully")