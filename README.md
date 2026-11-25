# Realtime_elt_pipeline_project

Real-time clickstream feature engineering and ML inference pipeline on AWS (Kinesis, Lambda, DynamoDB, FastAPI, CDK). Features 1-min time-bucketed sliding windows with atomic counters, partial batch failure handling via SQS DLQ, and sub-50ms online inference.

```
producers ──> Kinesis ──> Lambda consumer ──> DynamoDB (user_id, window_min)
                              │                       │
                              └── poison pills ──> SQS DLQ
                                                      │
                     FastAPI /predict/{user_id} <─────┘  (Query last N buckets -> merge -> model)
```

## Layout

| Path | Purpose |
| --- | --- |
| `shared/features.py` | Single source of truth for validation, minute bucketing, sliding-window merge, and model input vector. Used by Lambda, API, and training. |
| `shared/model.py` | Dependency-free logistic model (JSON artifact) so serving needs no numpy/sklearn. |
| `lambda_fn/lambda_function.py` | Kinesis consumer: validates, aggregates per `(user_id, minute)`, writes one atomic `UpdateItem` per bucket, routes poison pills to the DLQ, returns `batchItemFailures`. |
| `api/app.py` | FastAPI inference service plus `/dashboard`. |
| `infra/` | AWS CDK (Python) stack. |
| `scripts/train_model.py` | Time-split training of a first-purchase model using the shared feature code. |
| `scripts/load_test.py` | Throughput ramp, API latency/freshness probe, and DLQ verification. |
| `tests/` | Offline unit tests (no AWS required). |

## Data model

DynamoDB table, PK `user_id` (S), SK `window_min` (N = `epoch_seconds // 60`), on-demand billing, TTL on `expires_at` (bucket start + 1 hour). Each item holds `total_events`, `view_count`, `click_count`, `purchase_count`, `last_event_type`, `last_event_ts`, `updated_at_ms`, and `max_seq`.

Features served for the last 5 minutes (configurable 1-60 via `lookback_minutes`): `view_count_5m`, `click_count_5m`, `purchase_count_5m`, `total_events_5m`, `ctr_5m`, `events_per_min_5m`, `recency_sec`, plus `feature_freshness_ms` (now minus newest `updated_at_ms`). `purchase_count_5m` is reported but **excluded from the model** to avoid label leakage in a first-purchase model.

## Correctness guarantees

- **Atomic counters**: `ADD` updates, never read-modify-write, so concurrent shards and invocations cannot lose increments.
- **Idempotent replays**: each bucket stores the highest Kinesis sequence number applied (`max_seq`). Writes are conditional on `max_seq < min(batch seqs)`. On conflict (a replay after partial failure) the handler reads the item, keeps only records with sequence numbers greater than `max_seq`, and retries with an optimistic `max_seq = :expected` condition. Duplicates are never double-counted. This relies on Kinesis ordering within a shard (enforced by partitioning on `user_id`).
- **Out-of-order events**: counters always accumulate. `last_event_*` only moves forward in event time.
- **Poison pills** (bad base64, bad UTF-8, invalid JSON, schema violations) go to SQS with the raw payload and sequence number and are treated as processed, so they never block the shard. If the DLQ send itself fails, that record is reported as a batch item failure and retried.
- **Partial batch failure**: a throttled/failed bucket reports only its own sequence numbers. The event source mapping also has bisect-on-error, 5 retries, a 1 hour max record age, and an on-failure destination to the same DLQ.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                          # 43 offline tests
python scripts/train_model.py      # writes models/model.json
```

The included `models/model.json` is a baseline trained on synthetic data (time-based split, test AUC ~0.78). Retrain on real events with `--events events.jsonl` and optionally upload with `--s3-uri`.

### Deploy

```bash
npm install -g aws-cdk
cdk bootstrap        # once per account/region
cdk deploy           # outputs StreamName, TableName, DlqUrl, ConsumerFunctionName
```

The Lambda asset contains only `lambda_fn/` and `shared/` (dotfiles and caches excluded). It uses Python 3.12 on arm64.

### Run the API

```bash
export AWS_REGION=us-east-1 TABLE_NAME=<TableName>
# optional: MODEL_S3_URI=s3://bucket/models/model.json (otherwise models/model.json)
# optional demo buttons: ENABLE_DEMO_ENDPOINTS=true STREAM_NAME=<StreamName>
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `GET /predict/{user_id}?lookback_minutes=5` returns probability, features, `inference_latency_ms`, `feature_freshness_ms`. Each call also logs a CloudWatch Embedded Metric Format line (`RealtimeFeaturePipeline` namespace).
- `GET /dashboard` polls predictions every second.
- `POST /demo/burst/{user_id}` and `POST /demo/poison/{user_id}` are **off by default** (404) and should never be enabled on a public endpoint.

The API needs `dynamodb:Query` on the table (plus `s3:GetObject` for `MODEL_S3_URI` and `kinesis:PutRecord(s)` if demo endpoints are on).

### Load test and DLQ check

```bash
python scripts/load_test.py --stream <StreamName> --rates 100,500,1200 --duration 30 --api-url http://localhost:8000
python scripts/load_test.py --stream <StreamName> --verify-dlq --table <TableName> --dlq-url <DlqUrl>
```

The DLQ check sends 99 valid events and 1 malformed record for a fresh user, then confirms DynamoDB counts exactly 99 and the poison pill reached SQS.

A single provisioned shard accepts about 1,000 records/s, so the 1,200 eps stage is expected to show throttling. That stage demonstrates the shard limit. Add shards or switch to on-demand mode for more headroom. Latency targets (sub-50 ms server-side inference) depend on running the API in the same region as the table.

## Alarms

Three CloudWatch alarms: Lambda errors >= 1 in 5 minutes, Kinesis iterator age >= 60 s, and DLQ visible messages >= 1. Attach an SNS action to get notifications.

## IAM (least privilege)

- Lambda: `dynamodb:UpdateItem` and `dynamodb:GetItem` (conflict path only) on the table, `sqs:SendMessage` on the DLQ, and Kinesis read actions on this stream (granted by the event source).
- No wildcard resources.
