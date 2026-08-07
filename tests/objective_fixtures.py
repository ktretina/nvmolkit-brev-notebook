from dataclasses import replace
import json

import numpy as np
from rdkit import Chem

from chemistry_workflow import EvidenceRecord, WorkflowPhase, WorkflowReport, WorkflowState
import objective_challenge
from objective_challenge import CANDIDATE_COUNT, ObjectiveCandidate, ObjectiveContext
from objective_challenge import TARGET_FRACTION
from objective_challenge import (
    TerminationReason,
    accepted_maxima,
    baseline_terminal_run,
    build_action_menu,
    evaluate_selected_swap,
    finalize_no_legal_swap,
    measure_panel,
    terminal_objective_run,
    build_objective_evidence,
)


class FakeTensor:
    def __init__(self, values, *, device="cpu"):
        self.values = np.asarray(values, dtype=float)
        self.device = device

    def cpu(self):
        return self

    def torch(self):
        return self

    def numpy(self):
        return self.values.copy()


class FakeGpuResult:
    def __init__(self, values, *, device="cpu"):
        self.tensor = FakeTensor(values, device=device)

    def torch(self):
        return self.tensor


class FakeOptimizationResult:
    def __init__(self):
        self.energies = FakeTensor([1.0, 2.0, 3.0])
        self.converged = FakeTensor([1, 1, 1])
        self.mol_indices = FakeTensor([0, 1, 2])
        self.conf_indices = FakeTensor([0, 0, 0])


def optimized_state(baseline_optimal: bool = False) -> WorkflowState:
    smiles = ("CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCF", "C1CC1")
    distance = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.80, dtype=float)
    np.fill_diagonal(distance, 0.0)
    if not baseline_optimal:
        distance[0, 1] = distance[1, 0] = 0.35
    conformer_molecules = []
    representative_records = []
    for index in range(3):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles[index]))
        molecule.AddConformer(Chem.Conformer(molecule.GetNumAtoms()))
        conformer_molecules.append(molecule)
        representative_records.append({
            "molecule_id": f"mol-{index}",
            "molecule_index": index,
            "source_row": index,
            "cluster_id": index,
            "generated_conformer_count": 1,
        })
    return WorkflowState(
        phase=WorkflowPhase.OPTIMIZED,
        records=[
            {"id": f"mol-{index}", "smiles": value, "source_row": index}
            for index, value in enumerate(smiles)
        ],
        molecules=[Chem.MolFromSmiles(value) for value in smiles],
        fingerprints=FakeGpuResult(
            np.zeros((CANDIDATE_COUNT, 32), dtype=np.int32), device="cuda:0"
        ),
        fingerprint_parameters=(2, 1024),
        similarity=FakeGpuResult(1.0 - distance),
        clusters=[[index] for index in range(CANDIDATE_COUNT)],
        cluster_cutoff=0.4,
        representative_records=representative_records,
        conformer_molecules=conformer_molecules,
        optimization_result=FakeOptimizationResult(),
        summaries={"fixture": {"status": "complete"}},
        embedding_parameters=(3, "largest_clusters_first", 3),
    )


def two_revision_context() -> ObjectiveContext:
    candidates = tuple(
        ObjectiveCandidate(
            molecule_id=f"candidate-{index}",
            molecule_index=index,
            source_row=index,
            cluster_id=index,
        )
        for index in range(CANDIDATE_COUNT)
    )
    distance = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.90, dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[0, 1] = distance[1, 0] = 0.30
    distance[2, 3] = distance[3, 2] = 0.40
    distance.setflags(write=False)
    return ObjectiveContext(
        candidates=candidates,
        baseline_ids=("candidate-0", "candidate-1", "candidate-2", "candidate-3"),
        baseline_score=0.30,
        benchmark_score=0.90,
        target_score=0.78,
        distance_matrix=distance,
    )


def context_from_distance(
    distance: np.ndarray,
    *,
    baseline_ids: tuple[str, str, str, str] = (
        "mol-0",
        "mol-1",
        "mol-2",
        "mol-3",
    ),
) -> ObjectiveContext:
    values = np.array(distance, dtype=np.float64, copy=True)
    values.setflags(write=False)
    candidates = tuple(
        ObjectiveCandidate(f"mol-{index}", index, index, index)
        for index in range(CANDIDATE_COUNT)
    )
    provisional = ObjectiveContext(
        candidates, baseline_ids, 0.0, 0.0, 0.0, values
    )
    baseline = objective_challenge.measure_panel(provisional, baseline_ids).score
    benchmark = objective_challenge.attainable_benchmark(provisional).score
    target = float(baseline + TARGET_FRACTION * (benchmark - baseline))
    return replace(
        provisional,
        baseline_score=float(baseline),
        benchmark_score=float(benchmark),
        target_score=target,
    )


def controlled_context(
    *,
    distances: dict[tuple[str, str], float],
    default_distance: float,
) -> ObjectiveContext:
    matrix = np.full(
        (CANDIDATE_COUNT, CANDIDATE_COUNT), default_distance, dtype=np.float64
    )
    np.fill_diagonal(matrix, 0.0)
    for (first_id, second_id), value in distances.items():
        first = int(first_id.removeprefix("mol-"))
        second = int(second_id.removeprefix("mol-"))
        matrix[first, second] = matrix[second, first] = value
    return context_from_distance(matrix)


def controlled_context_with_ranked_swaps() -> ObjectiveContext:
    """Four isolated legal improvements with known descending score keys."""
    distances = {("mol-0", "mol-1"): 0.10}
    for replacement, blocked, quality in (
        ("mol-4", "mol-0", 0.90),
        ("mol-5", "mol-0", 0.80),
        ("mol-6", "mol-1", 0.70),
        ("mol-7", "mol-1", 0.60),
    ):
        distances[(replacement, blocked)] = 0.10
        for baseline in ("mol-0", "mol-1", "mol-2", "mol-3"):
            if baseline != blocked:
                distances[(replacement, baseline)] = quality
    return controlled_context(distances=distances, default_distance=0.95)


def target_achieved_context() -> ObjectiveContext:
    """One-revision context matching the accepted live objective distance."""
    final_distance = 0.8374999910593033
    matrix = np.full(
        (CANDIDATE_COUNT, CANDIDATE_COUNT), final_distance, dtype=np.float64
    )
    np.fill_diagonal(matrix, 0.0)
    matrix[0, 1] = matrix[1, 0] = 0.76744
    return context_from_distance(matrix)


def controlled_context_without_improving_swaps() -> ObjectiveContext:
    distances = {}
    for first in range(4):
        for second in range(first + 1, 4):
            distances[(f"mol-{first}", f"mol-{second}")] = 0.50
    for first in range(4):
        for second in range(4, 8):
            distances[(f"mol-{first}", f"mol-{second}")] = 0.50
    return controlled_context(distances=distances, default_distance=0.90)


def controlled_context_with_action_count(count: int) -> ObjectiveContext:
    if type(count) is not int or not 0 <= count <= 3:
        raise ValueError("Fixture action count must be in [0, 3].")
    distances = {("mol-0", "mol-1"): 0.10}
    specifications = (
        ("mol-4", "mol-0", 0.90),
        ("mol-5", "mol-0", 0.80),
        ("mol-6", "mol-1", 0.70),
        ("mol-7", "mol-1", 0.10),
    )
    for index, (replacement, blocked, quality) in enumerate(specifications):
        effective_quality = quality if index < count else 0.10
        distances[(replacement, blocked)] = 0.10
        for baseline in ("mol-0", "mol-1", "mol-2", "mol-3"):
            if baseline != blocked:
                distances[(replacement, baseline)] = effective_quality
    return controlled_context(distances=distances, default_distance=0.95)


def boundary_policy_context(candidate: float, current: float) -> ObjectiveContext:
    distances = {("mol-0", "mol-1"): current}
    for replacement in ("mol-4", "mol-5", "mol-6", "mol-7"):
        quality = candidate if replacement == "mol-4" else current
        distances[(replacement, "mol-0")] = current
        for baseline in ("mol-1", "mol-2", "mol-3"):
            distances[(replacement, baseline)] = quality
    return controlled_context(distances=distances, default_distance=0.95)


def controlled_context_with_tied_paths(all_paths_reach: bool) -> ObjectiveContext:
    distances = {
        ("mol-0", "mol-1"): 0.10,
        ("mol-4", "mol-0"): 0.10,
        ("mol-4", "mol-1"): 0.40,
        ("mol-5", "mol-0"): 0.40,
        ("mol-5", "mol-1"): 0.10,
        ("mol-6", "mol-0"): 0.10,
        ("mol-6", "mol-1"): 0.10,
        ("mol-6", "mol-5"): 0.10,
        ("mol-6", "mol-7"): 0.10,
        ("mol-4", "mol-7"): 0.10,
        ("mol-7", "mol-0"): 0.10,
        ("mol-7", "mol-1"): 0.10,
        ("mol-4", "mol-5"): 0.10,
        ("mol-4", "mol-6"): 0.90,
        ("mol-5", "mol-7"): 0.90 if all_paths_reach else 0.50,
    }
    return controlled_context(distances=distances, default_distance=0.90)


def controlled_context_with_three_misses() -> ObjectiveContext:
    matrix = np.array(
        [
            [0, .70, .05, .90, .75, .40, .90, .65],
            [.70, 0, .35, .15, .40, .75, .60, .50],
            [.05, .35, 0, .90, .35, .80, .95, .15],
            [.90, .15, .90, 0, .15, .80, .40, .20],
            [.75, .40, .35, .15, 0, .10, .60, .50],
            [.40, .75, .80, .80, .10, 0, .50, .35],
            [.90, .60, .95, .40, .60, .50, 0, .70],
            [.65, .50, .15, .20, .50, .35, .70, 0],
        ],
        dtype=np.float64,
    )
    return context_from_distance(matrix)


def quantized_baseline_target_context() -> ObjectiveContext:
    matrix = np.full((CANDIDATE_COUNT, CANDIDATE_COUNT), 0.5, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for first in range(4, 8):
        for second in range(first + 1, 8):
            matrix[first, second] = matrix[second, first] = 0.5000000000006
    return context_from_distance(matrix)


def terminal_fixture(reason: str | TerminationReason, attempt_count: int):
    reason = TerminationReason(reason)
    if reason is TerminationReason.BASELINE_ALREADY_OPTIMAL:
        if attempt_count != 0:
            raise ValueError("Baseline terminal fixture requires zero attempts.")
        state = optimized_state(baseline_optimal=True)
        return baseline_terminal_run(objective_challenge.build_objective_context(state))
    if reason is TerminationReason.NO_LEGAL_IMPROVING_SWAP:
        if attempt_count != 0:
            raise ValueError("No-legal terminal fixture requires zero attempts.")
        context = controlled_context_without_improving_swaps()
        current = measure_panel(context, context.baseline_ids)
        menu = build_action_menu(context, current, 0)
        return finalize_no_legal_swap(context, (), current, menu)

    context = (
        target_achieved_context()
        if reason is TerminationReason.TARGET_ACHIEVED
        else controlled_context_with_three_misses()
    )
    attempts = []
    current = measure_panel(context, context.baseline_ids)
    for number in range(1, attempt_count + 1):
        menu = build_action_menu(context, current, number - 1)
        attempt = evaluate_selected_swap(context, menu, accepted_maxima(menu)[0], number)
        attempts.append(attempt)
        current = attempt.measurement
    return terminal_objective_run(context, tuple(attempts), reason)


def evidence_report() -> WorkflowReport:
    """Return complete production-shaped E01-E06 evidence for finding tests."""
    representatives = [
        {"molecule_id": f"mol-{index}", "source_row": index, "cluster_id": index}
        for index in range(4)
    ]
    per_conformer_records = [
        {
            "molecule_id": f"mol-{molecule_index}",
            "cluster_id": molecule_index,
            "conformer_index": conformer_index,
            "energy_kcal_mol": float(-10 * (molecule_index + 1) + conformer_index),
            "converged": not (molecule_index == 3 and conformer_index == 4),
        }
        for molecule_index in range(4)
        for conformer_index in range(5)
    ]
    selected_conformer_records = [
        {
            **next(
                record
                for record in per_conformer_records
                if record["molecule_id"] == f"mol-{index}"
                and record["conformer_index"] == 0
            ),
            "selected_conformer_id": f"mol-{index}:conf-0",
        }
        for index in range(4)
    ]
    payloads = (
        ("E01", "Library inspection", {
            "raw_count": 256,
            "valid_count": 256,
            "invalid_count": 0,
            "invalid_ids": [],
            "preview_count": 24,
            "count_unit": "rows",
        }, "RDKit input validation"),
        ("E02", "Morgan fingerprints", {
            "fingerprint_radius": 2,
            "fingerprint_size_bits": 1024,
            "packed_shape": [256, 32],
            "molecule_count": 256,
            "active_bits_min": 3,
            "active_bits_median": 14.0,
            "active_bits_max": 37,
            "executor": "nvMolKit GPU",
            "size_unit": "bits",
        }, "MorganFingerprintGenerator"),
        ("E03", "Tanimoto similarity", {
            "matrix_shape": [256, 256],
            "q1": 0.071,
            "median": 0.118,
            "q3": 0.184,
            "p90": 0.291,
            "max_off_diagonal": 0.873,
            "most_similar_pair": {
                "molecule_ids": ["mol-17", "mol-203"],
                "source_rows": [17, 203],
                "similarity": 0.873,
            },
            "similarity_unit": "Tanimoto coefficient",
        }, "crossTanimotoSimilarity"),
        ("E04", "Fused Butina clusters", {
            "cutoff": 0.4,
            "cluster_count": 70,
            "singleton_count": 37,
            "singleton_fraction": 37 / 256,
            "largest_cluster_sizes": [15, 12, 11, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 7],
            "assignment_count": 256,
            "cutoff_unit": "Tanimoto distance",
        }, "fused_butina"),
        ("E05", "Representative embedding", {
            "requested_representative_count": 4,
            "selected_representative_count": 4,
            "selection_shortfall": 0,
            "representative_policy": "largest_clusters_first",
            "representatives": representatives,
            "requested_conformers_per_representative": 5,
            "generated_conformer_count": 20,
            "partial_embedding_ids": [],
            "zero_embedding_ids": [],
            "count_unit": "conformers",
        }, "EmbedMolecules"),
        ("E06", "MMFF94 optimization", {
            "attempted_conformer_count": 20,
            "converged_conformer_count": 19,
            "unconverged_conformer_count": 1,
            "per_conformer_records": per_conformer_records,
            "selected_conformer_records": selected_conformer_records,
            "energy_unit": "kcal/mol",
            "comparison_scope": "within molecule only",
        }, "MMFFOptimizeMoleculesConfs"),
    )
    return WorkflowReport(tuple(
        EvidenceRecord(
            key,
            label,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            provenance,
        )
        for key, label, payload, provenance in payloads
    ))


def report_and_run(
    reason: str | TerminationReason = TerminationReason.TARGET_ACHIEVED,
):
    reason = TerminationReason(reason)
    attempt_count = {
        TerminationReason.TARGET_ACHIEVED: 1,
        TerminationReason.ATTEMPT_LIMIT_REACHED: 3,
    }.get(reason, 0)
    run = terminal_fixture(reason, attempt_count)
    # Exercise Task 2's canonical terminal validator before returning a fixture.
    build_objective_evidence(run)
    return evidence_report(), run


def catalog(reason: str | TerminationReason = TerminationReason.TARGET_ACHIEVED):
    from objective_findings import build_finding_catalog

    report, run = report_and_run(reason)
    return build_finding_catalog(report, run)


def finding_selection(
    finding_catalog=None,
    *,
    reverse: bool = False,
):
    from objective_findings import CONCLUSION_THEMES

    finding_catalog = finding_catalog or catalog()
    ids = [finding_catalog.ids_for_theme(theme)[0] for theme in CONCLUSION_THEMES]
    if reverse:
        ids.reverse()
    return {"ordered_finding_ids": ids}


def completed_terminal_controller(client, reason=TerminationReason.TARGET_ACHIEVED):
    import demo_agent
    from chemistry_workflow import StageResult

    report, run = report_and_run(reason)
    accepted_count = len(run.attempts)
    session = demo_agent.AgentSession(
        messages=[
            {"role": "system", "content": "bounded chemistry agent"},
            {"role": "user", "content": "analyze"},
        ],
        state=optimized_state(),
        turn_count=7 + accepted_count,
    )
    plan = demo_agent.WorkflowPlan(stages=[
        {"stage": stage, "rationale": f"Run {stage}."}
        for stage in demo_agent.STAGES
    ])
    return demo_agent.BoundedWorkflowController(
        session=session,
        client=client,
        executors={},
        plan=plan,
        stage_results=[
            StageResult(stage, stage, {}) for stage in demo_agent.STAGES
        ],
        report=report,
        objective_required=True,
        objective_context=run.context,
        objective_attempts=list(run.attempts),
        accepted_attempt_count=accepted_count,
        selection_response_count=accepted_count,
        provider_request_attempt_count=accepted_count,
        objective_run=run,
        objective_evidence=build_objective_evidence(run),
        objective_prompt_appended=True,
    )


BOUNDARY_CASES = (
    (0.5000000000004, 0.5, False),
    (0.5000000000005, 0.5, True),
    (0.5000000000006, 0.5, True),
)


TARGET_BOUNDARY_CASES = (
    (0.4999999999994, 0.5, False),
    (0.4999999999995, 0.5, True),
    (0.5, 0.5, True),
    (0.5000000000005, 0.5, True),
)
