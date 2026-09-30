from src.evaluation import citation_coverage, reciprocal_rank, recall_at_k, token_f1


def test_recall_at_5():
    assert recall_at_k(["a", "b", "c", "d", "e"], ["b", "e"]) == 1.0
    assert recall_at_k(["a", "b"], ["b", "e"]) == 0.5


def test_mrr():
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_token_f1():
    score = token_f1("ICS uses common terminology", "ICS uses common terminology")
    assert score == 1.0
    assert 0 < token_f1("common terminology", "ICS uses common terminology") < 1


def test_citation_coverage():
    answer = "The EOC coordinates resources. [FEMA EOC Basics — Purpose and setup]"
    assert citation_coverage(answer, ["[FEMA EOC Basics — Purpose and setup]"], answerable=True) == 1.0
    assert citation_coverage("The documents do not contain enough evidence to answer this question.", [], answerable=False) == 1.0
