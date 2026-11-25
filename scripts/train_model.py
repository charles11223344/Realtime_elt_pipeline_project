"""Train a baseline first-purchase conversion model with a time-based split.

Features are produced by the SAME shared/features.py code used in Lambda and
the API (minute buckets -> sliding window), so there is no training-serving skew.

Usage:
    python scripts/train_model.py                       # synthetic data -> models/model.json
    python scripts/train_model.py --events events.jsonl # real events (one JSON per line)
    python scripts/train_model.py --s3-uri s3://bucket/models/model.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from shared.features import (  # noqa: E402
    DEFAULT_LOOKBACK_MINUTES,
    MODEL_FEATURES,
    aggregate_batch_by_bucket,
    build_sliding_window_features,
    to_model_input,
    validate_event,
)
from shared.model import LogisticModel  # noqa: E402

LABEL_HORIZON_SEC = 600
SAMPLE_EVERY_SEC = 120


def synthesize_events(n_users: int, duration_sec: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    start = 1_700_000_000
    events = []
    for u in range(n_users):
        intent = rng.betavariate(1.2, 6)  # latent purchase intent
        rate = 0.005 + 0.15 * intent  # events/sec
        t = start + rng.uniform(0, duration_sec * 0.3)
        while t < start + duration_sec:
            t += rng.expovariate(rate)
            r = rng.random()
            if r < 0.001 + 0.08 * intent * intent:
                etype = "purchase"
            elif r < 0.15 + 0.45 * intent:
                etype = "click"
            else:
                etype = "view"
            events.append({"user_id": f"user-{u}", "event_type": etype, "timestamp": int(t)})
    return events


def build_dataset(events: list[dict], lookback: int):
    by_user: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_user[e["user_id"]].append(e)

    rows = []
    for user_events in by_user.values():
        user_events.sort(key=lambda e: e["timestamp"])
        first_purchase = next(
            (e["timestamp"] for e in user_events if e["event_type"] == "purchase"), None
        )
        t0, t_end = user_events[0]["timestamp"], user_events[-1]["timestamp"]
        for now in range(t0 + 60, t_end, SAMPLE_EVERY_SEC):
            if first_purchase is not None and now >= first_purchase:
                break  # first-purchase model: stop sampling once the user converted
            seen = [
                (str(i), e)
                for i, e in enumerate(user_events)
                if now - lookback * 60 - 60 <= e["timestamp"] <= now
            ]
            buckets = list(aggregate_batch_by_bucket(seen).values())
            for b in buckets:
                b["updated_at_ms"] = b["last_event_ts"] * 1000
            features = build_sliding_window_features(buckets, now, lookback)
            label = int(first_purchase is not None and now < first_purchase <= now + LABEL_HORIZON_SEC)
            rows.append((now, to_model_input(features, lookback), label))

    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, help="JSONL file of raw events")
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--duration-sec", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lookback-minutes", type=int, default=DEFAULT_LOOKBACK_MINUTES)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "model.json")
    parser.add_argument("--s3-uri", help="Optional s3://bucket/key to upload the artifact")
    args = parser.parse_args()

    if args.events:
        events = []
        for line in args.events.read_text().splitlines():
            if line.strip():
                try:
                    events.append(validate_event(json.loads(line)))
                except ValueError:
                    continue
    else:
        events = synthesize_events(args.users, args.duration_sec, args.seed)

    rows = build_dataset(events, args.lookback_minutes)
    split = int(len(rows) * args.train_fraction)
    if split == 0 or split == len(rows):
        raise SystemExit("Not enough samples for a time-based split")
    split_ts = rows[split][0]
    train = [r for r in rows if r[0] < split_ts]
    test = [r for r in rows if r[0] >= split_ts]

    x_train = np.array([r[1] for r in train])
    y_train = np.array([r[2] for r in train])
    x_test = np.array([r[1] for r in test])
    y_test = np.array([r[2] for r in test])

    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1.0
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit((x_train - means) / scales, y_train)

    model = LogisticModel.from_dict(
        {
            "version": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "feature_names": list(MODEL_FEATURES),
            "coefficients": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "means": means.tolist(),
            "scales": scales.tolist(),
        }
    )

    probs = [model.predict_proba(v) for v in x_test.tolist()]
    auc = roc_auc_score(y_test, probs) if len(set(y_test.tolist())) > 1 else float("nan")
    print(
        json.dumps(
            {
                "train_samples": len(train),
                "test_samples": len(test),
                "train_positive_rate": round(float(y_train.mean()), 4),
                "test_positive_rate": round(float(y_test.mean()), 4),
                "test_auc": round(float(auc), 4),
            },
            indent=2,
        )
    )

    payload = json.dumps(model.to_dict(), indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(f"Wrote {args.output}")

    if args.s3_uri:
        import boto3
        from urllib.parse import urlparse

        parsed = urlparse(args.s3_uri)
        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=payload.encode(),
            ContentType="application/json",
        )
        print(f"Uploaded {args.s3_uri}")


if __name__ == "__main__":
    main()
