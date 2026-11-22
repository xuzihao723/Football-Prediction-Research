import pandas as pd
from src.pooled_split import legal_time,temporal_batches,assert_temporal

def test_date_only_legal_after_full_day():
    d=pd.DataFrame({'match_datetime':['2020-01-01 00:00:00+00:00'],'timestamp_precision':['date_only']})
    assert str(legal_time(d).iloc[0]).startswith('2020-01-02')

def test_same_timestamp_batched():
    d=pd.DataFrame({'match_datetime':['2020-01-01 12:00:00+00:00','2020-01-01 12:00:00+00:00','2020-01-02 12:00:00+00:00'],'timestamp_precision':['exact']*3})
    b=list(temporal_batches(d)); assert len(b)==2 and list(b[0])==[0,1]

def test_temporal_strict():
    tr=pd.DataFrame({'match_datetime':['2020-01-01 12:00:00+00:00'],'timestamp_precision':['exact']})
    te=pd.DataFrame({'match_datetime':['2020-01-02 12:00:00+00:00'],'timestamp_precision':['exact']})
    assert_temporal(tr,te)

def test_date_only_forces_whole_calendar_day_batch():
    d=pd.DataFrame({
        'match_datetime':['2020-01-01 09:00:00+00:00','2020-01-01 18:00:00+00:00','2020-01-02 12:00:00+00:00'],
        'timestamp_precision':['date_only','datetime','datetime']
    })
    b=list(temporal_batches(d))
    assert len(b)==2
    assert list(b[0])==[0,1]
