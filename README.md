# Introvert vs Extrovert Personality Prediction API

## Project Overview

This project predicts whether a person is an **Introvert** or an **Extrovert** using a Machine Learning model trained on personality-related features.

The application is built with **FastAPI** and exposes a REST API that accepts user input through a POST request and returns the predicted personality.

---

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Logistic Regression
- Pandas
- NumPy
- Joblib
- Uvicorn

---

## Machine Learning Workflow

- Data Exploration (EDA)
- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Feature Encoding
- Feature Scaling using StandardScaler
- Model Training
- Model Evaluation
- Model Deployment using FastAPI

---

## Project Structure

```
Introvert-Extrovert-Predictor/
│
├── api/
│   └── main.py
│
├── data/
│   └── personality_dataset.csv
│
├── models/
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── notebook/
│   └── Personality_Prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000/docs
```

---

## Sample Request

```json
{
  "Time_spent_Alone": 8,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2,
  "Going_outside": 3,
  "Drained_after_socializing": "No",
  "Friends_circle_size": 4,
  "Post_frequency": 2
}
```

---

## Sample Response

```json
{
  "prediction": "Extrovert"
}
```

---

## Author

**Rashmika Sewmini Rupasinghe**

AI/ML Internship Technical Assessment
