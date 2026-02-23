from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_register_and_login():
    r = client.post(
        "/api/auth/register",
        json={"email": "admin@example.com", "password": "secret123", "tenant_name": "wiztric", "role": "admin"},
    )
    assert r.status_code in (200, 400)
    r = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "secret123"},
    )
    assert r.status_code == 200
    token = r.json()["access_token"]
    assert token

