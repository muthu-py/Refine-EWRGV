from scripts.benchmark_eval.metrics import evaluate_ranking, aggregate


def test_trec_style_ranking_metrics_and_unjudged_convention():
    row = evaluate_ranking(["a", "x", "b", "b"], {"a": 2, "b": 1, "c": 0}, cutoff=10)
    assert row["P@5"] == 0.4
    assert row["P@10"] == 0.2
    assert row["R@10"] == 1.0
    assert row["RR"] == 1.0
    assert row["AP"] == (1.0 + 2 / 3) / 2
    assert row["nDCG@10"] > 0.0
    assert row["judged_retrieved@10"] == 2
    assert row["unjudged_retrieved@10"] == 1


def test_empty_relevant_set_and_macro_aggregation():
    empty = evaluate_ranking(["u"], {"j": 0})
    assert empty["RR"] == empty["AP"] == empty["R@10"] == 0.0
    macro = aggregate([empty, evaluate_ranking(["j"], {"j": 1})])
    assert macro["query_count"] == 2
    assert macro["RR"] == 0.5
