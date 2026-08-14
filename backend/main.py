from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:3000",
    "https://diabatese-predictor.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Get the folder where main.py is located
BASE_DIR = Path(__file__).resolve().parent
# Load our saved ML models
data = joblib.load(BASE_DIR / "models.pkl")
trained_models = data["models"]
model_accuracy = data["accuracy"]
best_model_name = data["best_model"]

@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running"
    }
from pydantic import BaseModel
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float


@app.post("/predict")
def predict(data: DiabetesInput):

    input_data = [[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]]

    predictions = {}

    for name, model in trained_models.items():

        prediction = model.predict(input_data)[0]

        predictions[name] = (
            "Diabetic" if prediction == 1 else "Not Diabetic"
        )

    return {
        "predictions": predictions,
        "accuracy": {
            name: round(acc * 100, 2)
            for name, acc in model_accuracy.items()
        },
        "best_model": best_model_name
    }