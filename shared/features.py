"""Pure feature transformations.

This module must stay free of boto3 / network calls so the exact same code runs
in unit tests, the Lambda stream consumer, the FastAPI inference service, and
offline training. That is what prevents training-serving skew.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

VALID_EVENT_TYPES = frozenset({"view", "click", "purchase"})
COUNTER_FIELDS = ("total_events", "view_count", "click_count", "purchase_count")
BUCKET_SECONDS = 60
BUCKET_TTL_SECONDS = 3600
DEFAULT_LOOKBACK_MINUTES = 5
MAX_USER_ID_LENGTH = 128

# purchase_count_5m is deliberately excluded: the model predicts first-purchase
# conversion, so including it would leak the label into the features.
MODEL_FEATURES = (
    "view_count_5m",
    "click_count_5m",
    "total_events_5m",
    "ctr_5m",
    "events_per_min_5m",
    "recency_sec",
)


def validate_event(event: Any) -> dict:
    """Return a normalized event or raise ValueError for malformed payloads."""
    if not isinstance(event, Mapping):
        raise ValueError("event must be a JSON object")

    user_id = event.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if len(user_id) > MAX_USER_ID_LENGTH:
        raise ValueError(f"user_id must be at most {MAX_USER_ID_LENGTH} characters")

    event_type = event.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(VALID_EVENT_TYPES)}")

    timestamp = event.get("timestamp")
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError("timestamp must be a positive Unix epoch integer (seconds)")

    return {"user_id": user_id, "event_type": event_type, "timestamp": timestamp}


def get_window_minute(timestamp: int | float) -> int:
    return int(timestamp) // BUCKET_SECONDS


def bucket_expires_at(window_min: int) -> int:
    return window_min * BUCKET_SECONDS + BUCKET_TTL_SECONDS


def _empty_bucket(user_id: str, window_min: int) -> dict:
    return {
        "user_id": user_id,
        "window_min": window_min,
        "total_events": 0,
        "view_count": 0,
        "click_count": 0,
        "purchase_count": 0,
        "last_event_type": None,
        "last_event_ts": 0,
        "sequence_numbers": [],
    }


def aggregate_batch_by_bucket(
    records: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[tuple[str, int], dict]:
    """Group validated events by (user_id, window_min) in memory.

    `records` is an iterable of (sequence_number, validated_event) pairs. The
    result collapses N events into one DynamoDB write per bucket and keeps the
    sequence numbers so a failed bucket can be retried precisely.
    """
    buckets: dict[tuple[str, int], dict] = {}
    for sequence_number, event in records:
        user_id = event["user_id"]
        ts = int(event["timestamp"])
        key = (user_id, get_window_minute(ts))
        agg = buckets.get(key)
        if agg is None:
            agg = buckets[key] = _empty_bucket(*key)

        agg["total_events"] += 1
        agg[f"{event['event_type']}_count"] += 1
        agg["sequence_numbers"].append(sequence_number)

        # Out-of-order arrivals must not overwrite a newer "last event".
        if ts >= agg["last_event_ts"]:
            agg["last_event_ts"] = ts
            agg["last_event_type"] = event["event_type"]
    return buckets


def build_sliding_window_features(
    buckets: Iterable[Mapping[str, Any]],
    now_ts: int | float,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
) -> dict:
    """Merge minute buckets into sliding-window features.

    Bucket values may be DynamoDB Decimals, so everything is coerced to int.
    """
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")

    cutoff = now_ts - lookback_minutes * BUCKET_SECONDS
    totals = dict.fromkeys(COUNTER_FIELDS, 0)
    last_event_ts = 0
    last_event_type = None
    latest_updated_ms = 0
    active = 0

    for bucket in buckets:
        bucket_last_ts = int(bucket.get("last_event_ts") or 0)
        if bucket_last_ts < cutoff:
            continue
        active += 1
        for field in COUNTER_FIELDS:
            totals[field] += int(bucket.get(field) or 0)
        if bucket_last_ts >= last_event_ts:
            last_event_ts = bucket_last_ts
            last_event_type = bucket.get("last_event_type")
        latest_updated_ms = max(latest_updated_ms, int(bucket.get("updated_at_ms") or 0))

    has_activity = active > 0
    return {
        "view_count_5m": totals["view_count"],
        "click_count_5m": totals["click_count"],
        "purchase_count_5m": totals["purchase_count"],
        "total_events_5m": totals["total_events"],
        "ctr_5m": totals["click_count"] / max(totals["view_count"], 1),
        "events_per_min_5m": totals["total_events"] / lookback_minutes,
        "recency_sec": max(0.0, float(now_ts - last_event_ts)) if has_activity else None,
        "last_event_type": last_event_type,
        "last_event_ts": last_event_ts if has_activity else None,
        "feature_freshness_ms": (
            max(0, latest_updated_ms - last_event_ts * 1000)
            if has_activity and latest_updated_ms
            else None
        ),
        "active_buckets": active,
        "lookback_minutes": lookback_minutes,
    }


def to_model_input(
    features: Mapping[str, Any],
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
) -> list[float]:
    """Build the ordered, leakage-free numeric vector the model consumes."""
    window_seconds = float(lookback_minutes * BUCKET_SECONDS)
    vector = []
    for name in MODEL_FEATURES:
        value = features.get(name)
        if name == "recency_sec":
            # No activity in the window is treated as "as stale as the window".
            value = window_seconds if value is None else min(float(value), window_seconds)
        vector.append(float(value or 0.0))
    return vector
