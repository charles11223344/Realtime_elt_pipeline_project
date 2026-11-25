"""Kinesis throughput + API latency benchmark, and DLQ verification.

Ramp mode (default): for each target rate, send batched put_records calls
concurrently for --duration seconds and report achieved throughput, throttling,
and put_records latency. Optionally probe the API for latency and freshness.

    python scripts/load_test.py --stream <name> --rates 100,500,1200 --duration 30 \
        --api-url http://localhost:8000

DLQ mode: send 99 valid events + 1 malformed record for a fresh user, then
verify DynamoDB counts 99 and the DLQ received the poison pill.

    python scripts/load_test.py --stream <name> --verify-dlq --table <t> --dlq-url <url>
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config

MAX_RECORDS_PER_CALL = 500


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def make_event(user_id: str, now: int) -> dict:
    etype = random.choices(["view", "click", "purchase"], weights=[70, 27, 3])[0]
    return {"user_id": user_id, "event_type": etype, "timestamp": now}


def put_chunk(kinesis, stream: str, records: list[dict]) -> dict:
    started = time.perf_counter()
    try:
        resp = kinesis.put_records(StreamName=stream, Records=records)
    except kinesis.exceptions.ProvisionedThroughputExceededException:
        return {"latency_ms": (time.perf_counter() - started) * 1000, "failed": len(records), "throttled": len(records)}
    latency_ms = (time.perf_counter() - started) * 1000
    throttled = sum(
        1 for r in resp["Records"] if r.get("ErrorCode") == "ProvisionedThroughputExceededException"
    )
    return {"latency_ms": latency_ms, "failed": resp.get("FailedRecordCount", 0), "throttled": throttled}


def run_stage(kinesis, stream: str, rate: int, duration: int, users: int, workers: int) -> dict:
    results = []
    user_ids = [f"load-{i}" for i in range(users)]
    stage_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for second in range(duration):
            tick = stage_start + second
            now = int(time.time())
            records = []
            for _ in range(rate):
                uid = random.choice(user_ids)
                records.append(
                    {"Data": json.dumps(make_event(uid, now)).encode(), "PartitionKey": uid}
                )
            chunks = [records[i : i + MAX_RECORDS_PER_CALL] for i in range(0, len(records), MAX_RECORDS_PER_CALL)]
            futures = [pool.submit(put_chunk, kinesis, stream, c) for c in chunks]
            results.extend(f.result() for f in futures)
            sleep_for = tick + 1 - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    elapsed = time.perf_counter() - stage_start
    sent = rate * duration
    failed = sum(r["failed"] for r in results)
    latencies = [r["latency_ms"] for r in results]
    return {
        "target_eps": rate,
        "sent": sent,
        "accepted": sent - failed,
        "failed": failed,
        "throttled": sum(r["throttled"] for r in results),
        "achieved_eps": round((sent - failed) / elapsed, 1),
        "put_records_p50_ms": round(percentile(latencies, 50), 1),
        "put_records_p99_ms": round(percentile(latencies, 99), 1),
    }


def probe_api(api_url: str, user_id: str, samples: int) -> dict:
    latencies, server_latencies, freshness = [], [], []
    for _ in range(samples):
        started = time.perf_counter()
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/predict/{user_id}", timeout=10) as resp:
            body = json.loads(resp.read())
        latencies.append((time.perf_counter() - started) * 1000)
        server_latencies.append(body["inference_latency_ms"])
        if body.get("feature_freshness_ms") is not None:
            freshness.append(body["feature_freshness_ms"])
    return {
        "client_mean_ms": round(statistics.mean(latencies), 1),
        "client_p99_ms": round(percentile(latencies, 99), 1),
        "server_mean_ms": round(statistics.mean(server_latencies), 2),
        "server_p99_ms": round(percentile(server_latencies, 99), 2),
        "feature_freshness_mean_ms": round(statistics.mean(freshness), 1) if freshness else None,
    }


def verify_dlq(kinesis, stream: str, table_name: str, dlq_url: str, wait_sec: int) -> dict:
    user_id = f"dlq-test-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    marker = f"POISON-{uuid.uuid4().hex[:8]}"
    records = [
        {"Data": json.dumps({"user_id": user_id, "event_type": "view", "timestamp": now}).encode(), "PartitionKey": user_id}
        for _ in range(99)
    ]
    records.insert(50, {"Data": f'{{"user_id": "{user_id}", "{marker}"'.encode(), "PartitionKey": user_id})
    resp = kinesis.put_records(StreamName=stream, Records=records)
    if resp.get("FailedRecordCount"):
        raise SystemExit(f"put_records rejected {resp['FailedRecordCount']} records; retry later")

    table = boto3.resource("dynamodb").Table(table_name)
    sqs = boto3.client("sqs")
    deadline = time.time() + wait_sec
    total, found_marker = 0, False
    while time.time() < deadline and not (total == 99 and found_marker):
        items = table.query(KeyConditionExpression=Key("user_id").eq(user_id), ConsistentRead=True)["Items"]
        total = sum(int(i.get("total_events", 0)) for i in items)
        msgs = sqs.receive_message(QueueUrl=dlq_url, MaxNumberOfMessages=10, WaitTimeSeconds=2).get("Messages", [])
        for m in msgs:
            if marker in m["Body"]:
                found_marker = True
                sqs.delete_message(QueueUrl=dlq_url, ReceiptHandle=m["ReceiptHandle"])
        time.sleep(1)
    return {"user_id": user_id, "dynamodb_total_events": total, "poison_in_dlq": found_marker, "passed": total == 99 and found_marker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--rates", default="100,500,1200", help="Comma-separated events/sec stages")
    parser.add_argument("--duration", type=int, default=30, help="Seconds per stage")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--api-url")
    parser.add_argument("--api-samples", type=int, default=200)
    parser.add_argument("--verify-dlq", action="store_true")
    parser.add_argument("--table")
    parser.add_argument("--dlq-url")
    parser.add_argument("--wait-sec", type=int, default=60)
    args = parser.parse_args()

    kinesis = boto3.client(
        "kinesis",
        config=Config(max_pool_connections=args.workers * 2, retries={"max_attempts": 1, "mode": "standard"}),
    )

    if args.verify_dlq:
        if not (args.table and args.dlq_url):
            parser.error("--verify-dlq requires --table and --dlq-url")
        print(json.dumps(verify_dlq(kinesis, args.stream, args.table, args.dlq_url, args.wait_sec), indent=2))
        return

    for rate in (int(r) for r in args.rates.split(",")):
        stage = run_stage(kinesis, args.stream, rate, args.duration, args.users, args.workers)
        if args.api_url:
            stage["api"] = probe_api(args.api_url, "load-0", args.api_samples)
        print(json.dumps(stage), flush=True)


if __name__ == "__main__":
    main()
