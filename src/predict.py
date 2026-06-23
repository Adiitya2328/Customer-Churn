import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Example customer
sample_customer = [[
    0,  # SeniorCitizen
    12, # tenure
    70.5, # MonthlyCharges
    850.0, # TotalCharges
    1,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0
]]

prediction = model.predict(sample_customer)

print("Prediction:", prediction)