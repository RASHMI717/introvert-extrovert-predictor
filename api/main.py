import utils.transport
import utils.session
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Create FastAPI Application
app = FastAPI()

# --------------------------------------------------
# Load Trained Model and Scaler
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Input Data Blueprint
# --------------------------------------------------

class PersonalityData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float

# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Introvert-Extrovert Personality Prediction API!"
    }

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

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
    personality = "Introvert" if prediction[0] == 0 else "Extrovert"

    return {
        "prediction": personality
    }