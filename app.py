from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('./models/notebooks/best_xgb_model.pkl')

class PredictionInput(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PredictionInput):
    input_features = np.array(data.features).reshape(1, -1)
    
    prediction = model.predict(input_features)
    
    return {"prediction": prediction.tolist()}

