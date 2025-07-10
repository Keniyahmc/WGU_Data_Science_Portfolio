#!/usr/bin/env python
# coding: utf-8

from fastapi.testclient import TestClient
from api import app  # Make sure this matches your actual API file name if different

client = TestClient(app)

# Test 1: Root endpoint returns status and message
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running"}

# Test 2: Valid prediction request with numeric airport codes (as used in airport_encodings.json)
def test_valid_prediction():
    response = client.get("/predict/delays", params={
        "departure_airport": "10397",  # ATL
        "arrival_airport": "12451",   # JFK
        "departure_time": 800,
        "arrival_time": 1000
    })
    assert response.status_code == 200
    assert "average_departure_delay_minutes" in response.json()

# Test 3: Missing departure_time param should return 422
def test_missing_parameters():
    response = client.get("/predict/delays", params={
        "departure_airport": "10397",
        "arrival_airport": "12451",
        "arrival_time": 1000  # Missing departure_time
    })
    assert response.status_code == 422  # Validation error
