"""Kinesis -> DynamoDB stream consumer.

- Poison pills (bad JSON / schema) go to the SQS DLQ and are NOT retried.
- Valid events are pre-aggregated per (user_id, minute) and written with
  atomic ADD counters, one update_item per bucket in the common case.
- Transient write failures return only that bucket's sequence numbers in
  batchItemFailures.

Idempotency: on partial failure Kinesis replays from the *lowest* failed
sequence number, which also replays buckets that already succeeded. Each
bucket therefore stores `max_seq` (the highest Kinesis sequence number applied,
zero-padded for lexicographic comparison). Writes are conditional on every
record being newer than `max_seq`, so a replay can never double-count. Because
records are partitioned by user_id, all records for a bucket share one shard
and their sequence numbers are totally ordered.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

from shared.features import aggregate_batch_by_bucket, bucket_expires_at, validate_event

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

TABLE_NAME = os.environ.get("TABLE_NAME", "")
DLQ_URL = os.environ.get("DLQ_URL", "")

# Created once per execution environment and reused across invocations.
dynamodb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")
table = dynamodb.Table(TABLE_NAME) if TABLE_NAME else None

MAX_DLQ_PAYLOAD_CHARS = 200_000
SEQ_WIDTH = 128  # Kinesis sequence numbers are numeric strings of up to 128 digits.
MAX_CONFLICT_RETRIES = 3

_COUNTER_ADD = (
    " ADD total_events :total, view_count :views, click_count :clicks, purchase_count :purchases"
)
_SET_META = "SET updated_at_ms = :updated, expires_at = :expires, max_seq = :max_seq"
_SET_LAST = ", last_event_type = :last_type, last_event_ts = :last_ts"


def pad_seq(sequence_number: str) -> str:
    return sequence_number.zfill(SEQ_WIDTH)


def _is_conditional_failure(err: ClientError) -> bool:
    return err.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


def _send_to_dlq(record: dict, raw_payload: str, reason: str) -> bool:
    kinesis = record.get("kinesis", {})
    body = {
        "reason": reason,
        "raw_payload": raw_payload[:MAX_DLQ_PAYLOAD_CHARS],
        "sequence_number": kinesis.get("sequenceNumber"),
        "partition_key": kinesis.get("partitionKey"),
        "event_source_arn": record.get("eventSourceARN"),
        "received_at_ms": int(time.time() * 1000),
    }
    try:
        sqs.send_message(QueueUrl=DLQ_URL, MessageBody=json.dumps(body))
        return True
    except ClientError:
        logger.exception("Failed to send poison pill to DLQ")
        return False


def _update_values(agg: dict, now_ms: int) -> dict:
    return {
        ":total": agg["total_events"],
        ":views": agg["view_count"],
        ":clicks": agg["click_count"],
        ":purchases": agg["purchase_count"],
        ":updated": now_ms,
        ":expires": bucket_expires_at(agg["window_min"]),
        ":max_seq": max(pad_seq(s) for s in agg["sequence_numbers"]),
    }


def _write_bucket(agg: dict, events_by_seq: dict[str, dict], now_ms: int) -> None:
    key = {"user_id": agg["user_id"], "window_min": agg["window_min"]}

    # Fast path: one write. Fails if any record was already applied (replay) or
    # if a newer last_event is already stored (out-of-order across batches).
    try:
        table.update_item(
            Key=key,
            UpdateExpression=_SET_META + _SET_LAST + _COUNTER_ADD,
            ConditionExpression=(
                "(attribute_not_exists(max_seq) OR max_seq < :min_seq) AND "
                "(attribute_not_exists(last_event_ts) OR last_event_ts <= :last_ts)"
            ),
            ExpressionAttributeValues={
                **_update_values(agg, now_ms),
                ":min_seq": min(pad_seq(s) for s in agg["sequence_numbers"]),
                ":last_type": agg["last_event_type"],
                ":last_ts": agg["last_event_ts"],
            },
        )
        return
    except ClientError as err:
        if not _is_conditional_failure(err):
            raise

    # Slow path: read the stored watermark, drop already-applied records, and
    # write again guarded by optimistic concurrency on max_seq.
    for _ in range(MAX_CONFLICT_RETRIES):
        stored = table.get_item(
            Key=key,
            ConsistentRead=True,
            ProjectionExpression="max_seq, last_event_ts",
        ).get("Item", {})
        stored_max_seq = stored.get("max_seq")
        stored_last_ts = int(stored.get("last_event_ts") or 0)

        pending = [
            (seq, events_by_seq[seq])
            for seq in agg["sequence_numbers"]
            if stored_max_seq is None or pad_seq(seq) > stored_max_seq
        ]
        if not pending:
            return  # Entire bucket was already applied by a previous attempt.

        fresh = next(iter(aggregate_batch_by_bucket(pending).values()))
        values = _update_values(fresh, now_ms)
        expression = _SET_META
        if fresh["last_event_ts"] >= stored_last_ts:
            expression += _SET_LAST
            values[":last_type"] = fresh["last_event_type"]
            values[":last_ts"] = fresh["last_event_ts"]

        if stored_max_seq is None:
            condition = "attribute_not_exists(max_seq)"
        else:
            condition = "max_seq = :expected_seq"
            values[":expected_seq"] = stored_max_seq

        try:
            table.update_item(
                Key=key,
                UpdateExpression=expression + _COUNTER_ADD,
                ConditionExpression=condition,
                ExpressionAttributeValues=values,
            )
            return
        except ClientError as err:
            if not _is_conditional_failure(err):
                raise
    raise RuntimeError(f"Exceeded conflict retries for bucket {key}")


def lambda_handler(event, context):
    records = event.get("Records", [])
    failures: list[str] = []
    valid: list[tuple[str, dict]] = []
    poison_count = 0

    for record in records:
        sequence_number = record["kinesis"]["sequenceNumber"]
        raw_payload = ""
        try:
            raw_payload = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
            valid.append((sequence_number, validate_event(json.loads(raw_payload))))
        except (ValueError, UnicodeDecodeError, TypeError) as err:
            # json.JSONDecodeError subclasses ValueError. Never retry these.
            poison_count += 1
            if not _send_to_dlq(record, raw_payload, f"{type(err).__name__}: {err}"):
                # Losing a poison pill silently is worse than retrying it once.
                failures.append(sequence_number)

    events_by_seq = dict(valid)
    buckets = aggregate_batch_by_bucket(valid)
    now_ms = int(time.time() * 1000)
    for agg in buckets.values():
        try:
            _write_bucket(agg, events_by_seq, now_ms)
        except (ClientError, RuntimeError):
            logger.exception(
                "DynamoDB write failed for user_id=%s window_min=%s",
                agg["user_id"],
                agg["window_min"],
            )
            failures.extend(agg["sequence_numbers"])

    logger.info(
        json.dumps(
            {
                "records": len(records),
                "valid": len(valid),
                "poison": poison_count,
                "buckets": len(buckets),
                "failed_records": len(failures),
            }
        )
    )
    return {"batchItemFailures": [{"itemIdentifier": seq} for seq in failures]}
