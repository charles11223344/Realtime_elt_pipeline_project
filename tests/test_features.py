from decimal import Decimal

import pytest

from shared.features import (
    MODEL_FEATURES,
    aggregate_batch_by_bucket,
    bucket_expires_at,
    build_sliding_window_features,
    get_window_minute,
    to_model_input,
    validate_event,
)

BASE = 1_700_000_040  # exactly on a minute boundary (28333334 * 60)


def ev(user="u1", etype="view", ts=BASE):
    return {"user_id": user, "event_type": etype, "timestamp": ts}


class TestValidateEvent:
    def test_valid_event_is_normalized(self):
        assert validate_event({**ev(), "extra": 1}) == ev()

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            "str",
            {"event_type": "view", "timestamp": BASE},
            {"user_id": "", "event_type": "view", "timestamp": BASE},
            {"user_id": "   ", "event_type": "view", "timestamp": BASE},
            {"user_id": 123, "event_type": "view", "timestamp": BASE},
            {"user_id": "x" * 129, "event_type": "view", "timestamp": BASE},
            ev(etype="scroll"),
            ev(etype=None),
            ev(ts=0),
            ev(ts=-5),
            ev(ts=1.5),
            ev(ts="1700000000"),
            ev(ts=True),
            {"user_id": "u1", "event_type": "view"},
        ],
    )
    def test_malformed_events_raise(self, payload):
        with pytest.raises(ValueError):
            validate_event(payload)


class TestBuckets:
    def test_minute_boundaries(self):
        assert get_window_minute(BASE) == BASE // 60
        assert get_window_minute(BASE + 59) == BASE // 60
        assert get_window_minute(BASE + 60) == BASE // 60 + 1
        assert get_window_minute(BASE - 1) == BASE // 60 - 1

    def test_expires_at_is_one_hour_after_bucket_start(self):
        assert bucket_expires_at(100) == 100 * 60 + 3600

    def test_aggregates_per_user_and_minute(self):
        records = [
            ("1", ev("u1", "view", BASE)),
            ("2", ev("u1", "click", BASE + 30)),
            ("3", ev("u1", "view", BASE + 60)),
            ("4", ev("u2", "purchase", BASE + 10)),
        ]
        buckets = aggregate_batch_by_bucket(records)
        assert len(buckets) == 3
        b = buckets[("u1", BASE // 60)]
        assert (b["total_events"], b["view_count"], b["click_count"], b["purchase_count"]) == (2, 1, 1, 0)
        assert b["sequence_numbers"] == ["1", "2"]
        assert buckets[("u2", BASE // 60)]["purchase_count"] == 1

    def test_out_of_order_does_not_regress_last_event(self):
        records = [
            ("1", ev(etype="click", ts=BASE + 40)),
            ("2", ev(etype="view", ts=BASE + 5)),  # arrives later, happened earlier
        ]
        b = aggregate_batch_by_bucket(records)[("u1", BASE // 60)]
        assert b["last_event_ts"] == BASE + 40
        assert b["last_event_type"] == "click"
        assert b["total_events"] == 2


class TestSlidingWindow:
    def _bucket(self, minute_offset, last_ts, views=0, clicks=0, purchases=0, updated_ms=None):
        total = views + clicks + purchases
        return {
            "window_min": Decimal(BASE // 60 + minute_offset),
            "total_events": Decimal(total),
            "view_count": Decimal(views),
            "click_count": Decimal(clicks),
            "purchase_count": Decimal(purchases),
            "last_event_type": "click",
            "last_event_ts": Decimal(last_ts),
            "updated_at_ms": Decimal(updated_ms if updated_ms is not None else last_ts * 1000 + 250),
        }

    def test_merges_active_buckets_and_drops_stale(self):
        now = BASE + 300
        buckets = [
            self._bucket(-1, BASE - 30, views=10, clicks=5),  # older than 5 min -> dropped
            self._bucket(0, BASE + 50, views=4, clicks=2),
            self._bucket(4, BASE + 290, views=4, clicks=2, purchases=1, updated_ms=(BASE + 290) * 1000 + 800),
        ]
        f = build_sliding_window_features(buckets, now, 5)
        assert f["active_buckets"] == 2
        assert f["view_count_5m"] == 8
        assert f["click_count_5m"] == 4
        assert f["purchase_count_5m"] == 1
        assert f["total_events_5m"] == 13
        assert f["ctr_5m"] == pytest.approx(0.5)
        assert f["events_per_min_5m"] == pytest.approx(13 / 5)
        assert f["recency_sec"] == 10
        assert f["feature_freshness_ms"] == 800

    def test_ctr_with_no_views_does_not_divide_by_zero(self):
        f = build_sliding_window_features([self._bucket(0, BASE, clicks=3)], BASE + 1, 5)
        assert f["ctr_5m"] == 3.0

    def test_decays_to_zero_after_inactivity(self):
        f = build_sliding_window_features([self._bucket(0, BASE, views=3)], BASE + 301, 5)
        assert f["total_events_5m"] == 0
        assert f["recency_sec"] is None
        assert f["feature_freshness_ms"] is None

    def test_freshness_never_negative(self):
        b = self._bucket(0, BASE, views=1, updated_ms=BASE * 1000 - 500)
        assert build_sliding_window_features([b], BASE + 1, 5)["feature_freshness_ms"] == 0

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            build_sliding_window_features([], BASE, 0)


class TestModelInput:
    def test_excludes_purchase_count(self):
        assert "purchase_count_5m" not in MODEL_FEATURES

    def test_vector_order_and_empty_recency(self):
        f = build_sliding_window_features([], BASE, 5)
        vec = to_model_input(f, 5)
        assert len(vec) == len(MODEL_FEATURES)
        assert vec[MODEL_FEATURES.index("recency_sec")] == 300.0
        assert all(isinstance(v, float) for v in vec)
