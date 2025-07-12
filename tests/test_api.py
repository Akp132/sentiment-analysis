import requests, json, os

BASE = os.getenv("BASE_URL", "http://localhost:8000")

def test_predict_smoke():
    r = requests.post(f"{BASE}/predict", json={"text": "Great product!"})
    assert r.status_code == 200
    payload = r.json()
    assert set(payload) == {"label", "score"}
    assert isinstance(payload["score"], float) 