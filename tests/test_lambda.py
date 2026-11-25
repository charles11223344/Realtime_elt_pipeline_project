"""Offline tests for the Lambda handler using an in-memory fake DynamoDB table."""

import base64
import json

import pytest
from botocore.exceptions import ClientError

from lambda_fn import lambda_function as lf

BASE = 1_700_000_040


def _err(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "UpdateItem")


class FakeTable:
    """Implements just enough update_item/get_item semantics for the handler."""

    def __init__(self, throttle_users=()):
        self.items = {}
        self.throttle_users = set(throttle_users)

    def get_item(self, Key, **_):
        item = self.items.get((Key["user_id"], Key["window_min"]))
        return {"Item": dict(item)} if item else {}

    def update_item(self, Key, UpdateExpression, ExpressionAttributeValues, ConditionExpression=None):
        if Key["user_id"] in self.throttle_users:
            raise _err("ProvisionedThroughputExceededException")
        k = (Key["user_id"], Key["window_min"])
        item = self.items.get(k, {})
        v = ExpressionAttributeValues
        if ConditionExpression and not self._condition(ConditionExpression, item, v):
            raise _err("ConditionalCheckFailedException")
        new = dict(item)
        for f, p in [("total_events", ":total"), ("view_count", ":views"), ("click_count", ":clicks"), ("purchase_count", ":purchases")]:
            new[f] = new.get(f, 0) + v[p]
        new.update(updated_at_ms=v[":updated"], expires_at=v[":expires"], max_seq=v[":max_seq"])
        if "last_event_ts = :last_ts" in UpdateExpression:
            new.update(last_event_type=v[":last_type"], last_event_ts=v[":last_ts"])
        self.items[k] = new

    @staticmethod
    def _condition(expr, item, v):
        if expr == "attribute_not_exists(max_seq)":
            return "max_seq" not in item
        if expr == "max_seq = :expected_seq":
            return item.get("max_seq") == v[":expected_seq"]
        seq_ok = "max_seq" not in item or item["max_seq"] < v[":min_seq"]
        ts_ok = "last_event_ts" not in item or item["last_event_ts"] <= v[":last_ts"]
        return seq_ok and ts_ok


class FakeSqs:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def send_message(self, QueueUrl, MessageBody):
        if self.fail:
            raise _err("ServiceUnavailable")
        self.messages.append(json.loads(MessageBody))


def record(seq, payload):
    data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    if isinstance(data, str):
        data = data.encode()
    return {"kinesis": {"sequenceNumber": str(seq), "partitionKey": "pk", "data": base64.b64encode(data).decode()}}


def ev(user="u1", etype="view", ts=BASE):
    return {"user_id": user, "event_type": etype, "timestamp": ts}


@pytest.fixture
def fakes(monkeypatch):
    table, sqs = FakeTable(), FakeSqs()
    monkeypatch.setattr(lf, "table", table)
    monkeypatch.setattr(lf, "sqs", sqs)
    monkeypatch.setattr(lf, "DLQ_URL", "https://sqs.test/dlq")
    return table, sqs


def total(table, user="u1"):
    return sum(i["total_events"] for (u, _), i in table.items.items() if u == user)


def test_one_poison_pill_among_99_valid(fakes):
    table, sqs = fakes
    records = [record(i, ev()) for i in range(1, 100)]
    records.insert(50, record(1000, '{"user_id": "u1", broken'))
    resp = lf.lambda_handler({"Records": records}, None)
    assert resp == {"batchItemFailures": []}
    assert total(table) == 99
    assert len(sqs.messages) == 1
    assert "broken" in sqs.messages[0]["raw_payload"]
    assert sqs.messages[0]["sequence_number"] == "1000"


def test_schema_violation_and_bad_utf8_go_to_dlq(fakes):
    table, sqs = fakes
    resp = lf.lambda_handler({"Records": [record(1, ev(etype="scroll")), record(2, b"\xff\xfe")]}, None)
    assert resp["batchItemFailures"] == []
    assert len(sqs.messages) == 2
    assert table.items == {}


def test_dlq_send_failure_is_retried(fakes, monkeypatch):
    monkeypatch.setattr(lf, "sqs", FakeSqs(fail=True))
    resp = lf.lambda_handler({"Records": [record(7, "not json")]}, None)
    assert resp["batchItemFailures"] == [{"itemIdentifier": "7"}]


def test_throttled_bucket_reports_only_its_sequence_numbers(fakes):
    table, _ = fakes
    table.throttle_users = {"u2"}
    records = [record(1, ev("u1")), record(2, ev("u2")), record(3, ev("u1")), record(4, ev("u2"))]
    resp = lf.lambda_handler({"Records": records}, None)
    assert sorted(f["itemIdentifier"] for f in resp["batchItemFailures"]) == ["2", "4"]
    assert total(table, "u1") == 2


def test_replay_after_partial_failure_does_not_double_count(fakes):
    table, _ = fakes
    first = [record(1, ev("u1")), record(2, ev("u2")), record(3, ev("u1"))]
    table.throttle_users = {"u2"}
    lf.lambda_handler({"Records": first}, None)
    # Kinesis replays from the lowest failed sequence number (2), which includes u1's record 3.
    table.throttle_users = set()
    replay = first[1:] + [record(4, ev("u1"))]
    assert lf.lambda_handler({"Records": replay}, None) == {"batchItemFailures": []}
    assert total(table, "u1") == 3  # records 1, 3, 4 exactly once
    assert total(table, "u2") == 1


def test_full_duplicate_batch_is_noop(fakes):
    table, _ = fakes
    batch = [record(1, ev()), record(2, ev(etype="click"))]
    lf.lambda_handler({"Records": batch}, None)
    lf.lambda_handler({"Records": batch}, None)
    assert total(table) == 2


def test_out_of_order_across_batches_keeps_latest_last_event(fakes):
    table, _ = fakes
    lf.lambda_handler({"Records": [record(1, ev(etype="click", ts=BASE + 50))]}, None)
    lf.lambda_handler({"Records": [record(2, ev(etype="view", ts=BASE + 10))]}, None)
    item = table.items[("u1", BASE // 60)]
    assert item["total_events"] == 2
    assert item["last_event_ts"] == BASE + 50
    assert item["last_event_type"] == "click"


def test_bucket_is_written_once_per_batch_with_ttl(fakes, monkeypatch):
    table, _ = fakes
    calls = []
    original = table.update_item
    monkeypatch.setattr(table, "update_item", lambda **kw: (calls.append(kw), original(**kw)))
    lf.lambda_handler({"Records": [record(i, ev()) for i in range(1, 51)]}, None)
    assert len(calls) == 1
    assert table.items[("u1", BASE // 60)]["expires_at"] == (BASE // 60) * 60 + 3600
