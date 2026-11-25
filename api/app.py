"""Low-latency online inference service.

Run locally:
    TABLE_NAME=... AWS_REGION=... uvicorn api.app:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.dynamodb.conditions import Key
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import HTMLResponse

from shared.features import (
    DEFAULT_LOOKBACK_MINUTES,
    MAX_USER_ID_LENGTH,
    build_sliding_window_features,
    get_window_minute,
    to_model_input,
)
from shared.model import LogisticModel

logger = logging.getLogger("inference")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
DEFAULT_MODEL_PATH = ROOT / "models" / "model.json"


def load_model() -> LogisticModel:
    s3_uri = os.environ.get("MODEL_S3_URI")
    if s3_uri:
        parsed = urlparse(s3_uri)
        body = boto3.client("s3").get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return LogisticModel.from_json(body["Body"].read())
    return LogisticModel.from_json(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).read_bytes())


@asynccontextmanager
async def lifespan(app: FastAPI):
    table_name = os.environ.get("TABLE_NAME")
    if not table_name:
        raise RuntimeError("TABLE_NAME environment variable is required")
    # Long-lived clients: connection pools are reused across requests.
    app.state.table = boto3.resource("dynamodb").Table(table_name)
    app.state.model = load_model()
    stream_name = os.environ.get("STREAM_NAME")
    demo_enabled = os.environ.get("ENABLE_DEMO_ENDPOINTS", "false").lower() == "true"
    app.state.stream_name = stream_name if demo_enabled else None
    app.state.kinesis = boto3.client("kinesis") if app.state.stream_name else None
    logger.info("Loaded model version=%s table=%s", app.state.model.version, table_name)
    yield


app = FastAPI(title="Realtime Feature Inference", lifespan=lifespan)

UserId = PathParam(..., min_length=1, max_length=MAX_USER_ID_LENGTH, pattern=r"^[A-Za-z0-9_\-:.]+$")


def query_buckets(table, user_id: str, now_ts: float, lookback_minutes: int) -> list[dict]:
    start_min = get_window_minute(now_ts) - (lookback_minutes - 1)
    items: list[dict] = []
    kwargs = {
        "KeyConditionExpression": Key("user_id").eq(user_id) & Key("window_min").gte(start_min)
    }
    while True:
        page = table.query(**kwargs)
        items.extend(page.get("Items", []))
        if "LastEvaluatedKey" not in page:
            return items
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model.version}


@app.get("/predict/{user_id}")
def predict(
    request: Request,
    user_id: str = UserId,
    lookback_minutes: int = Query(DEFAULT_LOOKBACK_MINUTES, ge=1, le=60),
):
    started = time.perf_counter()
    now_ts = time.time()
    buckets = query_buckets(request.app.state.table, user_id, now_ts, lookback_minutes)
    features = build_sliding_window_features(buckets, now_ts, lookback_minutes)
    probability = request.app.state.model.predict_proba(to_model_input(features, lookback_minutes))
    latency_ms = (time.perf_counter() - started) * 1000

    # CloudWatch Embedded Metric Format: becomes metrics when shipped via CloudWatch Logs.
    print(
        json.dumps(
            {
                "_aws": {
                    "Timestamp": int(now_ts * 1000),
                    "CloudWatchMetrics": [
                        {
                            "Namespace": "RealtimeFeaturePipeline",
                            "Dimensions": [[]],
                            "Metrics": [
                                {"Name": "inference_latency_ms", "Unit": "Milliseconds"},
                                {"Name": "feature_freshness_ms", "Unit": "Milliseconds"},
                            ],
                        }
                    ],
                },
                "inference_latency_ms": round(latency_ms, 3),
                "feature_freshness_ms": features["feature_freshness_ms"] or 0,
            }
        ),
        flush=True,
    )

    return {
        "user_id": user_id,
        "purchase_probability": round(probability, 6),
        "model_version": request.app.state.model.version,
        "features": features,
        "inference_latency_ms": round(latency_ms, 3),
        "feature_freshness_ms": features["feature_freshness_ms"],
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


def _require_demo(request: Request):
    if not request.app.state.kinesis:
        raise HTTPException(
            status_code=404,
            detail="Demo endpoints are disabled. Set ENABLE_DEMO_ENDPOINTS=true and STREAM_NAME.",
        )
    return request.app.state.kinesis, request.app.state.stream_name


@app.post("/demo/burst/{user_id}")
def demo_burst(request: Request, user_id: str = UserId, count: int = Query(50, ge=1, le=500)):
    kinesis, stream = _require_demo(request)
    now = int(time.time())
    types = random.choices(["view", "click", "purchase"], weights=[70, 27, 3], k=count)
    records = [
        {
            "Data": json.dumps({"user_id": user_id, "event_type": t, "timestamp": now}).encode(),
            "PartitionKey": user_id,
        }
        for t in types
    ]
    resp = kinesis.put_records(StreamName=stream, Records=records)
    return {"sent": count, "failed": resp.get("FailedRecordCount", 0)}


@app.post("/demo/poison/{user_id}")
def demo_poison(request: Request, user_id: str = UserId):
    kinesis, stream = _require_demo(request)
    payload = f'{{"user_id": "{user_id}", "event_type": "click", "timestamp": BROKEN-{uuid.uuid4().hex[:6]}'
    kinesis.put_record(StreamName=stream, Data=payload.encode(), PartitionKey=user_id)
    return {"sent": 1, "payload": payload}
