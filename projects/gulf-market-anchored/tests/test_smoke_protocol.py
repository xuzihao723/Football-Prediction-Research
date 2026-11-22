from src.run_experiments import _smoke_data
from src import hpo
from src.experiments.common import split_fold


def test_smoke_never_contains_definitive_seasons():
    d=_smoke_data()
    assert set(d.season.astype(str)) <= {'2015-2016','2016-2017','2017-2018','2018-2019','2019-2020'}
    hpo.set_smoke(True)
    try:
        tr,te=split_fold(d,0)
    finally:
        hpo.set_smoke(False)
    assert set(te.season.astype(str))=={'2019-2020'}
    assert set(tr.season.astype(str)) <= {'2015-2016','2016-2017','2017-2018','2018-2019'}
