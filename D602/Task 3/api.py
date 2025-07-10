#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and airport encodings
model = joblib.load("finalized_model.pkl")
with open("airport_encodings.json") as f:
    airport_encodings = json.load(f)

class DelayRequest(BaseModel):
    departure_airport: str
    arrival_airport: str
    departure_time: int
    arrival_time: int

@app.get("/")
def root():
    return {"message": "API is up and running"}

@app.get("/predict/delays")
def predict_delays(departure_airport: str, arrival_airport: str, departure_time: int, arrival_time: int):
    try:
        if arrival_airport.upper() not in airport_encodings:
            raise HTTPException(status_code=400, detail="Invalid arrival airport code")

        # One-hot encode the arrival airport
        num_airports = len(airport_encodings)
        airport_vector = np.zeros(num_airports)
        airport_index = airport_encodings[arrival_airport.upper()]
        airport_vector[airport_index] = 1

        # Include constant term as first feature (1.0)
        features = np.concatenate([[1.0, departure_time, arrival_time], airport_vector])

        # Ensure shape is correct (1, 61)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        return {"average_departure_delay_minutes": round(float(prediction), 2)}

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed. Check input format and model compatibility.")
