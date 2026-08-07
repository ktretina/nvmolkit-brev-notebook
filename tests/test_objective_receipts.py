import inspect
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from demo_agent import ObjectiveProposal
from objective_challenge import (
    ObjectiveAttempt, ObjectiveCandidate, ObjectiveContext, ObjectiveSwap,
    TARGET_FRACTION, evaluate_diverse_panel, rank_legal_swaps,
)
from objective_receipts import ObjectiveReceipt, objective_receipt


def proposal() -> ObjectiveProposal:
    return ObjectiveProposal(
        selected_ids=["mol-0", "mol-2", "mol-5", "mol-7"],
        decision_basis="Replace the limiting analogue.",
    )


def selected_swap() -> ObjectiveSwap:
    return ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.80,
        score_delta=0.80 - 0.35,
        limiting_pair=("mol-0", "mol-2"),
    )


def swapped_proposal() -> ObjectiveProposal:
    return ObjectiveProposal(
        selected_ids=["mol-0", "mol-5", "mol-2", "mol-3"],
        decision_basis="Replace the member of the measured limiting pair.",
    )


def prior_attempt() -> ObjectiveAttempt:
    return ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=0.35,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )


def evaluator_records():
    candidate_ids = tuple(f"mol-{index}" for index in range(8))
    distance = np.full((8, 8), 0.80, dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[0, 1] = distance[1, 0] = 0.35
    context = ObjectiveContext(
        candidates=tuple(
            ObjectiveCandidate(molecule_id, index, index, index)
            for index, molecule_id in enumerate(candidate_ids)
        ),
        baseline_ids=candidate_ids[:4],
        baseline_score=0.35,
        benchmark_score=0.80,
        target_score=0.35 + TARGET_FRACTION * (0.80 - 0.35),
        distance_matrix=distance,
    )
    first = evaluate_diverse_panel(
        context, context.baseline_ids, attempt_number=1,
        decision_basis="Measure the baseline.",
    )
    swap = rank_legal_swaps(context, first)[0]
    proposal = ObjectiveProposal(
        selected_ids=list(swap.resulting_ids),
        decision_basis="Use the ranked legal replacement.",
    )
    expected = evaluate_diverse_panel(
        context, tuple(proposal.selected_ids), attempt_number=2,
        decision_basis=proposal.decision_basis, selected_swap=swap,
    )
    return context, first, swap, proposal, expected


def test_objective_receipt_displays_validated_ids_and_fixed_executor():
    receipt = objective_receipt(proposal())

    assert receipt == ObjectiveReceipt(
        validated_proposal=(
            "select_diverse_panel(selected_ids=['mol-0', 'mol-2', 'mol-5', 'mol-7'])"
        ),
        validated_intervention=None,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    context,\n"
            "    tuple(proposal.selected_ids),\n"
            "    attempt_number=len(self.objective_attempts) + 1,\n"
            "    decision_basis=proposal.decision_basis,\n"
            "    selected_swap=self.pending_objective_swap,\n"
            ")"
        ),
    )


def test_objective_receipt_renders_exact_selected_intervention_without_scores_or_model_text():
    receipt = objective_receipt(swapped_proposal(), selected_swap(), prior_attempt())

    assert receipt.validated_intervention == (
        "replace molecule_id='mol-1' with molecule_id='mol-5'"
    )
    assert "0.80" not in receipt.validated_intervention
    assert "0.45" not in receipt.validated_intervention
    assert "limiting" not in receipt.validated_intervention
    assert "measured limiting pair" not in receipt.validated_intervention


def test_objective_receipt_accepts_one_score_key_improvement_below_old_tolerance():
    current = 0.5
    improved = 0.5000000000005
    delta = improved - current
    assert 0.0 < delta <= 1e-12
    swap = replace(
        selected_swap(), predicted_score=improved, score_delta=delta
    )
    prior = replace(prior_attempt(), score=current)

    receipt = objective_receipt(swapped_proposal(), swap, prior)

    assert receipt.validated_intervention is not None


def test_objective_receipt_rejects_inexact_raw_delta_even_within_old_tolerance():
    swap = selected_swap()
    forged_delta = float(np.nextafter(swap.score_delta, 1.0))

    with pytest.raises(ValueError, match="prior measurement"):
        objective_receipt(
            swapped_proposal(), replace(swap, score_delta=forged_delta), prior_attempt()
        )


def test_objective_receipt_python_evaluation_matches_controller_evaluator_signature():
    receipt = objective_receipt(proposal())
    signature = inspect.signature(evaluate_diverse_panel)
    keyword_names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert "    context," in receipt.python_evaluation
    assert "    tuple(proposal.selected_ids)," in receipt.python_evaluation
    assert "len(self.objective_attempts) + 1" in receipt.python_evaluation
    assert "self.pending_objective_swap" in receipt.python_evaluation
    assert {"attempt_number", "decision_basis", "selected_swap"} == keyword_names
    assert all(f"    {name}=" in receipt.python_evaluation for name in keyword_names)
    assert "candidate_pool" not in receipt.python_evaluation
    assert "similarity_matrix" not in receipt.python_evaluation


def test_objective_receipt_python_evaluation_executes_against_controller_like_state():
    context, first, swap, proposal, expected = evaluator_records()
    receipt = objective_receipt(proposal, swap, first)
    namespace = {
        "context": context,
        "proposal": proposal,
        "self": SimpleNamespace(
            objective_attempts=[first], pending_objective_swap=swap,
        ),
        "evaluate_diverse_panel": evaluate_diverse_panel,
    }

    exec(receipt.python_evaluation, namespace)

    assert namespace["result"] == expected


def test_objective_receipt_rejects_wrong_exact_swap_type():
    class SwapSubclass(ObjectiveSwap):
        pass

    with pytest.raises(ValueError, match="exact swap"):
        objective_receipt(
            swapped_proposal(), SwapSubclass(**selected_swap().__dict__), prior_attempt()
        )


def test_objective_receipt_requires_a_canonical_prior_for_revisions():
    class AttemptSubclass(ObjectiveAttempt):
        pass

    with pytest.raises(ValueError, match="prior"):
        objective_receipt(swapped_proposal(), selected_swap())
    with pytest.raises(ValueError, match="exact"):
        objective_receipt(
            swapped_proposal(), selected_swap(), AttemptSubclass(**prior_attempt().__dict__)
        )
    with pytest.raises(ValueError, match="prior"):
        objective_receipt(
            swapped_proposal(), selected_swap(), replace(prior_attempt(), achieved=True)
        )


@pytest.mark.parametrize(
    "swap",
    [
        ObjectiveSwap(
            replace_id="mol-1",
            replacement_id="mol-5",
            resulting_ids=("mol-0", "mol-5", "mol-2", "mol-7"),
            predicted_score=0.80,
            score_delta=0.45,
            limiting_pair=("mol-0", "mol-2"),
        ),
        ObjectiveSwap(
            replace_id="mol-1",
            replacement_id="mol-5",
            resulting_ids=("mol-0", "mol-1", "mol-5", "mol-2"),
            predicted_score=0.80,
            score_delta=0.45,
            limiting_pair=("mol-0", "mol-2"),
        ),
    ],
)
def test_objective_receipt_rejects_false_swap_panel_or_id_provenance(swap):
    with pytest.raises(ValueError, match="provenance"):
        objective_receipt(swapped_proposal(), swap, prior_attempt())


@pytest.mark.parametrize(
    "swap, prior",
    [
        (
            selected_swap(),
            replace(
                prior_attempt(),
                selected_ids=("mol-0", "mol-4", "mol-2", "mol-3"),
            ),
        ),
        (replace(selected_swap(), score_delta=0.40), prior_attempt()),
    ],
)
def test_objective_receipt_rejects_forged_predecessor_or_score_delta(swap, prior):
    with pytest.raises(ValueError, match="provenance"):
        objective_receipt(swapped_proposal(), swap, prior)


@pytest.mark.parametrize(
    "swap, prior",
    [
        (selected_swap(), replace(prior_attempt(), constraints_passed=False)),
        (selected_swap(), replace(prior_attempt(), score=float("nan"))),
        (
            replace(selected_swap(), predicted_score=0.35, score_delta=0.0),
            prior_attempt(),
        ),
        (
            replace(selected_swap(), predicted_score=float("inf")),
            prior_attempt(),
        ),
        (replace(selected_swap(), score_delta=float("nan")), prior_attempt()),
    ],
)
def test_objective_receipt_rejects_unmeasured_or_nonpositive_revision_provenance(swap, prior):
    with pytest.raises(ValueError):
        objective_receipt(swapped_proposal(), swap, prior)


def test_objective_receipt_excludes_decision_text_and_is_deterministic():
    first = objective_receipt(proposal())
    second = objective_receipt(proposal())

    assert first == second
    assert "Replace the limiting analogue" not in repr(first)


def test_objective_receipt_rejects_wrong_exact_type():
    class ProposalSubclass(ObjectiveProposal):
        pass

    with pytest.raises(ValueError, match="objective schema"):
        objective_receipt(ProposalSubclass(**proposal().model_dump()))


def test_objective_receipt_is_frozen():
    receipt = objective_receipt(proposal())
    with pytest.raises(FrozenInstanceError):
        receipt.validated_proposal = "changed"
