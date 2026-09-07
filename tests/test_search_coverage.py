"""Synthetic candidate universes with explicit, replayable exclusions."""

import json

import pytest

from buttcrack.search_coverage import SearchReceipt, audit_period_tuples, audit_word_bank


def test_word_bank_preserves_every_input_and_filter_reason():
    source = [" elm ", "ELM", "I", "", "silver", "blue-bird", "straße", "ash"]
    audit = audit_word_bank(source, min_length=3, max_length=5)
    assert audit.accepted == ("ELM", "ASH")
    assert [d.supplied for d in audit.decisions] == source
    assert audit.excluded_counts == {
        "duplicate": 1,
        "too_short": 1,
        "empty": 1,
        "too_long": 1,
        "invalid_symbols": 2,
    }
    assert json.loads(json.dumps(audit.to_dict()))["accepted"] == ["ELM", "ASH"]
    assert audit.sha256 == audit_word_bank(iter(source), min_length=3, max_length=5).sha256
    assert audit.sha256 != audit_word_bank(source + [""], min_length=3, max_length=5).sha256
    assert audit.sha256 != audit_word_bank(source, min_length=2, max_length=5).sha256


def test_period_geometry_and_order_are_explicit():
    source = [(2, 8), (3, 8), (8, 3), (3, 8)]
    audit = audit_period_tuples(source)
    assert audit.accepted == ((2, 8), (3, 8), (8, 3))
    restricted = audit_period_tuples(source, require_pairwise_coprime=True)
    assert restricted.accepted == ((3, 8), (8, 3))
    assert restricted.excluded_counts == {"not_pairwise_coprime": 1, "duplicate": 1}
    assert restricted.sha256 != audit.sha256


@pytest.mark.parametrize("periods", [[], [0], [-2], [True], [2.5]])
def test_invalid_periods_do_not_disappear_into_an_exclusion(periods):
    with pytest.raises(ValueError):
        audit_period_tuples([periods])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_length": 0},
        {"min_length": True},
        {"max_length": 0},
        {"min_length": 5, "max_length": 2},
        {"alphabet": "ABA"},
        {"alphabet": "ab"},
    ],
)
def test_bad_bank_filters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        audit_word_bank(["ASH"], **kwargs)


def test_receipt_links_exact_manifest_and_distinguishes_incomplete_work():
    audit = audit_word_bank(["ASH", "ELM", "FIR", "OAK"])
    receipt = SearchReceipt(audit, "timed_out", 1, ("fixed alphabet",), timeouts=1, capped=1)
    result = receipt.to_dict()
    assert result["audit"]["sha256"] == audit.sha256
    assert result["unattempted"] == 1
    assert not result["family_exclusion"]
    assert not result["proof_verified"]
    for counts in ({"evaluated": 3}, {"evaluated": 3, "timeouts": 1}):
        with pytest.raises(ValueError, match="all accepted candidates"):
            SearchReceipt(audit, "exhausted_bank", assumptions=("fixed alphabet",), **counts)
    complete = SearchReceipt(audit, "exhausted_bank", 4, ("fixed alphabet",))
    assert complete.to_dict()["unattempted"] == 0
    assert not complete.to_dict()["family_exclusion"]


def test_empty_bank_is_not_family_exclusion():
    audit = audit_word_bank(["I"], min_length=2)
    result = SearchReceipt(audit, "exhausted_bank", 0, ("minimum length two",)).to_dict()
    assert result["intended"] == 0
    assert result["audit"]["excluded_counts"] == {"too_short": 1}
    assert not result["family_exclusion"]


@pytest.mark.parametrize(
    "status,counts",
    [
        ("timed_out", {}),
        ("capped", {}),
        ("candidate_found", {}),
        ("exact_contradiction", {}),
        ("statistical_mismatch", {}),
        ("unknown", {}),
        ("interrupted", {"evaluated": -1}),
        ("interrupted", {"evaluated": True}),
        ("interrupted", {"evaluated": 1, "timeouts": 1}),
    ],
)
def test_invalid_outcome_accounting(status, counts):
    audit = audit_word_bank(["ASH"])
    counts.setdefault("evaluated", 0)
    with pytest.raises(ValueError):
        SearchReceipt(audit, status, assumptions=("fixed alphabet",), **counts)


def test_gate_failure_and_proof_attestation_do_not_claim_exclusion():
    audit = audit_period_tuples([(2, 8)])
    gate = SearchReceipt(audit, "gate_failed", 0, ("fixed periods",)).to_dict()
    assert gate["evaluated"] == 0
    assert gate["unattempted"] == 1
    proof = SearchReceipt(
        audit,
        "exact_contradiction",
        1,
        ("fixed periods and observed symbols",),
        proof_reference="local-proof.json",
    ).to_dict()
    assert proof["proof_reference"] == "local-proof.json"
    assert not proof["proof_verified"]
    assert not proof["family_exclusion"]
    with pytest.raises(ValueError, match="assumptions"):
        SearchReceipt(audit, "gate_failed", 0, ())


def test_contradiction_cannot_cover_unfinished_candidates():
    audit = audit_period_tuples([(2, 8), (3, 8)])
    with pytest.raises(ValueError, match="all accepted candidates"):
        SearchReceipt(
            audit,
            "exact_contradiction",
            1,
            ("fixed periods",),
            proof_reference="local-proof.json",
        )
    statistical = SearchReceipt(audit, "statistical_mismatch", 2, ("synthetic null model",))
    assert not statistical.to_dict()["family_exclusion"]
