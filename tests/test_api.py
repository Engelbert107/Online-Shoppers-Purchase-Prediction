from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_payload = {
    "Administrative": 1,
    "Administrative_Duration": 10.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 5,
    "ProductRelated_Duration": 120.0,
    "BounceRates": 0.01,
    "ExitRates": 0.02,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "May",
    "OperatingSystems": 2,
    "Browser": 1,
    "Region": 3,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json=[sample_payload])
    assert response.status_code == 200
    data = response.json()[0]
    assert "prediction" in data
    assert "probability" in data
