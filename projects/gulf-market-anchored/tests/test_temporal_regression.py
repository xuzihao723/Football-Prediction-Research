import pandas as pd

from src.pooled_split import temporal_batches


def test_batches_keep_input_order_with_duplicate_index_and_timezone_offsets():
    dates = pd.to_datetime([
        "2020-01-02 09:00:00+00:00",
        "2020-01-02 18:00:00+00:00",
        "2020-01-01 18:00:00-05:00",
        "2020-01-01 23:00:00+00:00",
        "2020-01-02 00:00:00+00:00",
        "2020-01-03 13:00:00+00:00",
    ], utc=True)
    fixtures = pd.DataFrame(
        {"match_datetime": dates, "timestamp_precision": ["exact", "exact", "exact",
         "exact", "date_only", "exact"]},
        index=[8, 8, 2, 3, 3, 2],
    )
    assert [group.tolist() for group in temporal_batches(fixtures)] == [[2, 3], [0, 1, 4], [5]]


def test_batches_use_date_fallback_without_precision_and_allow_empty_input():
    fixtures = pd.DataFrame({"date": ["2020-01-02", "2020-01-01", "2020-01-01"]})
    assert [group.tolist() for group in temporal_batches(fixtures)] == [[1, 2], [0]]
    assert list(temporal_batches(fixtures.iloc[:0])) == []
