import time
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from api import app as api_module
from shared.features import get_window_minute


class FakeTable:
    def __init__(self, items):
        self.items = items
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        return {"Items": self.items}


@pytest.fixture
def client(monkeypatch):
    now = time.time()
    minute = get_window_minute(now)
    table = FakeTable(
        [
            {
                "user_id": "u1",
                "window_min": Decimal(minute),
                "total_events": Decimal(6),
                "view_count": Decimal(4),
                "click_count": Decimal(2),
                "purchase_count": Decimal(0),
                "last_event_type": "click",
                "last_event_ts": Decimal(int(now) - 2),
                "updated_at_ms": Decimal(int(now * 1000) - 1500),
            }
        ]
    )
    monkeypatch.setenv("TABLE_NAME", "test-table")
    monkeypatch.delenv("ENABLE_DEMO_ENDPOINTS", raising=False)
    monkeypatch.setattr(api_module.boto3, "resource", lambda *_a, **_k: type("R", (), {"Table": lambda self, n: table})())
    with TestClient(api_module.app) as c:
        yield c, table


def test_health(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_returns_features_and_probability(client):
    c, table = client
    resp = c.get("/predict/u1")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["features"]["total_events_5m"] == 6
    assert body["features"]["ctr_5m"] == pytest.approx(0.5)
    assert body["feature_freshness_ms"] >= 0
    assert table.calls == 1


@pytest.mark.parametrize("bad", ["x" * 129, "has space", "semi;colon"])
def test_predict_rejects_bad_user_ids(client, bad):
    c, _ = client
    assert c.get(f"/predict/{bad}").status_code == 422


def test_demo_endpoints_disabled_by_default(client):
    c, _ = client
    assert c.post("/demo/burst/u1").status_code == 404
    assert c.post("/demo/poison/u1").status_code == 404


def test_dashboard_served(client):
    c, _ = client
    resp = c.get("/dashboard")
    assert resp.status_code == 200
    assert "Realtime Feature Pipeline" in resp.text
