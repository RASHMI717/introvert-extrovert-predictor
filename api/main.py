from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI Application

app = FastAPI()

# Load Trained Model and Scaler

model = joblib.load("../models/logistic_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Input Data Blueprint

class PersonalityData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float


# Home Endpoint

@app.get("/")
def home():
    return {
        "message": "Welcome to the Introvert-Extrovert Personality Prediction API!"
    }

# Prediction Endpoint

@app.post("/predict")
def predict(data: PersonalityData):

    # Convert categorical values into numerical values
    stage_fear = 1 if data.Stage_fear.lower() == "yes" else 0
    drained = 1 if data.Drained_after_socializing.lower() == "yes" else 0

    # Create DataFrame
    input_data = pd.DataFrame({
        "Time_spent_Alone": [data.Time_spent_Alone],
        "Stage_fear": [stage_fear],
        "Social_event_attendance": [data.Social_event_attendance],
        "Going_outside": [data.Going_outside],
        "Drained_after_socializing": [drained],
        "Friends_circle_size": [data.Friends_circle_size],
        "Post_frequency": [data.Post_frequency]
    })

    # Scale features
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Convert numerical prediction to label
    if prediction[0] == 0:
        personality = "Introvert"
    else:
        personality = "Extrovert"

    # Return result
    return {
        "prediction": personality
    }