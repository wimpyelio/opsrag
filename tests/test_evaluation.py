import pytest

from src.evaluation import (
    citation_coverage,
    evaluate_item,
    failure_category,
    recall_at_k,
    reciprocal_rank,
    summarize,
    token_f1,
)


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


# ── New tests added after stress-test findings ────────────────────────────────


# token_f1 edge cases ─────────────────────────────────────────────────────────

def test_token_f1_empty_prediction_is_zero():
    assert token_f1("", "hello world") == 0.0


def test_token_f1_empty_reference_is_zero():
    assert token_f1("hello world", "") == 0.0


def test_token_f1_both_empty_is_one():
    assert token_f1("", "") == 1.0


def test_token_f1_is_case_insensitive():
    assert token_f1("Hello World", "hello world") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("foo bar", "baz qux") == 0.0


# recall_at_k edge cases ──────────────────────────────────────────────────────

def test_recall_at_k_empty_retrieved_is_zero():
    assert recall_at_k([], ["a", "b"]) == 0.0


def test_recall_at_k_no_relevant_ids_is_trivially_one():
    # Convention: if there is nothing to find, retrieval trivially succeeds.
    assert recall_at_k(["a", "b"], []) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5


def test_recall_at_k_duplicate_relevant_ids_handled():
    # Deduplication via set() means 2x "a" collapses to 1 unique relevant.
    assert recall_at_k(["a"], ["a", "a"]) == 1.0


# reciprocal_rank edge cases ──────────────────────────────────────────────────

def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_not_found_is_zero():
    assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_reciprocal_rank_empty_retrieved_is_zero():
    assert reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_picks_earliest_of_multiple_relevant():
    # "a" is at rank 2, "b" is at rank 3 → MRR should reflect rank 2.
    result = reciprocal_rank(["x", "a", "b"], ["a", "b"])
    assert result == pytest.approx(0.5)


# citation_coverage edge cases ────────────────────────────────────────────────

def test_citation_coverage_partial():
    answer = "Answer. [Doc A — One]"
    result = citation_coverage(
        answer, ["[Doc A — One]", "[Doc B — Two]"], answerable=True
    )
    assert result == pytest.approx(0.5)


def test_citation_coverage_is_case_insensitive():
    answer = "Answer. [DOC A — ONE]"
    result = citation_coverage(answer, ["[Doc A — One]"], answerable=True)
    assert result == 1.0


def test_citation_coverage_unanswerable_all_refusal_markers():
    markers = [
        "not in the provided documents",
        "not contain enough evidence",
        "insufficient evidence",
        "cannot answer this question",
    ]
    for marker in markers:
        assert citation_coverage(marker, [], answerable=False) == 1.0, (
            f"Refusal marker not detected: {marker!r}"
        )


def test_citation_coverage_unanswerable_fabricated_answer_is_zero():
    result = citation_coverage(
        "ICS uses common terminology.", [], answerable=False
    )
    assert result == 0.0


def test_citation_coverage_answerable_empty_labels_is_zero():
    # No expected labels for an answerable question = cannot satisfy coverage.
    assert citation_coverage("Some answer.", [], answerable=True) == 0.0


# failure_category edge cases ─────────────────────────────────────────────────

def test_failure_category_none_on_perfect_result():
    assert failure_category(
        recall5=1.0, correctness=1.0, citation_cov=1.0, answerable=True
    ) == "none"


def test_failure_category_irrelevant_retrieval_also_sets_missing_evidence():
    # recall5=0 fires both labels (recall5=0 ⊂ recall5<1).
    result = failure_category(
        recall5=0.0, correctness=0.9, citation_cov=1.0, answerable=True
    )
    assert "irrelevant retrieval" in result
    assert "missing evidence" in result


def test_failure_category_unanswerable_only_checks_answer_level():
    result = failure_category(
        recall5=None, correctness=0.3, citation_cov=0.0, answerable=False
    )
    assert "unsupported answer" in result
    assert "irrelevant retrieval" not in result
    assert "missing evidence" not in result
    assert "wrong source" not in result


# summarize edge cases ────────────────────────────────────────────────────────

def test_summarize_averages_recall_over_answerable_subset_only():
    rows = [
        {"recall_at_5": 1.0, "mrr": 1.0, "answer_correctness": 0.8, "citation_coverage": 1.0},
        {"recall_at_5": 0.0, "mrr": 0.0, "answer_correctness": 0.4, "citation_coverage": 0.0},
        {"recall_at_5": "",  "mrr": "",  "answer_correctness": 1.0, "citation_coverage": 1.0},  # unanswerable
    ]
    s = summarize(rows)
    assert s["questions"] == 3
    assert s["answerable_questions"] == 2
    assert s["Recall@5"] == pytest.approx((1.0 + 0.0) / 2, abs=1e-4)
    assert s["MRR"] == pytest.approx((1.0 + 0.0) / 2, abs=1e-4)


# evaluate_item integration ───────────────────────────────────────────────────

def test_evaluate_item_unanswerable_question_omits_retrieval_metrics():
    item = {
        "id": "q035",
        "question": "How many radios are in the cache?",
        "reference_answer": "This question cannot be answered from the provided documents.",
        "relevant_chunks": [],
    }
    answer = "The provided documents do not contain enough evidence to answer this question."
    result = evaluate_item(item, [], answer)

    assert result["recall_at_5"] == ""
    assert result["mrr"] == ""


def test_evaluate_item_correct_retrieval_scores_full_recall():
    item = {
        "id": "q001",
        "question": "What is an EOC?",
        "reference_answer": "An EOC coordinates resources.",
        "relevant_chunks": ["eoc-c0001"],
    }
    retrieved = [{"chunk_id": "eoc-c0001", "title": "EOC", "section": "Basics"}]
    answer = "An EOC coordinates resources. [EOC — Basics]\n\nSources: [EOC — Basics]"

    result = evaluate_item(item, retrieved, answer)

    assert result["recall_at_5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["citation_coverage"] == 1.0
