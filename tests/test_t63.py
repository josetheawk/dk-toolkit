import random

from tools.dk.platforms.gc_mwcc import t63


def test_top3_share():
    v = {"regions": [{"differing": 5}, {"differing": 3}, {"differing": 1}, {"differing": 1}]}
    assert abs(t63.top3_share(v) - 0.9) < 1e-9
    assert t63.top3_share({"regions": []}) == 0.0


def test_stratified_spans_sizes():
    items = [{"size": i} for i in range(100)]
    got = t63.stratified(items, lambda x: x["size"], 10, random.Random(1))
    assert len(got) == 10
    assert min(g["size"] for g in got) < 10 and max(g["size"] for g in got) >= 90
