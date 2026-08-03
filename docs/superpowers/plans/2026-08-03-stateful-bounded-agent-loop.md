# Stateful Bounded nvMolKit Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six isolated Nemotron demonstrations with one persistent, bounded agent conversation that explains the fixed workflow, chooses evidence-informed parameters, invokes five visible nvMolKit entry points, and produces one mechanically grounded conclusion.

**Architecture:** `chemistry_workflow.py` owns deterministic RDKit/nvMolKit state, executors, figures, invariants, and evidence. `demo_agent.py` owns the single hosted conversation, strict schemas, dependency-valid calls, evidence-checked conclusion, and the `run_workflow(...)` facade. The notebook becomes a thin presentation surface with one user goal and one workflow invocation.

**Tech Stack:** CPython 3.12, Jupyter/nbformat, OpenAI-compatible NVIDIA hosted inference, Pydantic 2, nvMolKit 0.5.0, PyTorch CUDA, RDKit, pandas, NumPy, Matplotlib, seaborn, pytest.

---

## Execution Protocol

Use the user's selected subagent-driven workflow:

1. Create an isolated worktree and `codex/stateful-nvmolkit-agent` branch from the current local `main`.
2. Dispatch one fresh implementation agent for each numbered task.
3. After each task, dispatch a fresh specification-compliance reviewer.
4. After compliance passes, dispatch a fresh code-quality reviewer.
5. The primary agent resolves review findings, runs the task's focused tests, and records the commit before starting the next task.
6. Do not modify the surrounding BioNeMo Platform Meta Skill workbench.

Create the implementation worktree at execution time:

```bash
git worktree add /private/tmp/nvmolkit-stateful-agent -b codex/stateful-nvmolkit-agent main
```

Expected: a new worktree at `/private/tmp/nvmolkit-stateful-agent` on `codex/stateful-nvmolkit-agent`, containing specification commit `c3af9e9`.

Before every task:

```bash
git status --short --branch
```

Expected: only changes owned by the active task. Never add or remove the unrelated `.DS_Store` in the original checkout.

## File Map

- Create `chemistry_workflow.py`: deterministic workflow state, RDKit preprocessing, nvMolKit executors, representative selection, figures, report invariants, and evidence ledger.
- Modify `demo_agent.py`: persistent hosted conversation, strict plan/tool/conclusion schemas, tool-result history, eight-turn limit, and public workflow facade.
- Rewrite `notebooks/nvmolkit_nemotron_demo.ipynb`: no scientific function definitions; one user goal and one `run_workflow(...)` call.
- Create `tests/test_chemistry_workflow.py`: domain-state, representative, report, rendering, and failure tests.
- Rewrite `tests/test_demo_agent.py`: stateful conversation, eligible schemas, secret safety, and synthesis validation.
- Simplify `tests/test_notebook.py`: thin-notebook narrative and safety contract while retaining Launchable/setup health tests.
- Modify `README.md`: describe the bounded agent loop and live acceptance boundary.
- Modify `tests/test_gpu_acceptance.py` only to drive the new domain facade while preserving its five nvMolKit entry-point assertions. Keep `requirements.txt`, `launchable/`, `data/`, `skills/`, and `tests/test_skill_snapshot.py` unchanged.

### Task 1: Establish Deterministic Workflow State and Input Inspection

**Files:**
- Create: `chemistry_workflow.py`
- Create: `tests/test_chemistry_workflow.py`

- [ ] **Step 1: Write failing state and inspection tests**

Create tests that define the public domain contracts before implementation:

```python
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from chemistry_workflow import (
    StageResult,
    WorkflowPhase,
    WorkflowState,
    eligible_stage,
    inspect_library,
)


def test_new_state_exposes_only_input_inspection():
    state = WorkflowState()
    assert state.phase is WorkflowPhase.NEW
    assert eligible_stage(state) == "inspect_library"
    assert state.summaries == {}


def test_inspection_uses_fixed_columns_and_reports_invalid_rows(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {"id": "valid-1", "smiles": "CCO"},
            {"id": "invalid-1", "smiles": "not-smiles"},
            {"id": "valid-2", "smiles": "c1ccccc1"},
        ]
    ).to_csv(sample, index=False)
    state = WorkflowState()

    result = inspect_library(state, sample, expected_rows=3)

    assert isinstance(result, StageResult)
    assert state.phase is WorkflowPhase.INSPECTED
    assert result.summary == {
        "raw_count": 3,
        "valid_count": 2,
        "invalid_count": 1,
        "invalid_ids": ["invalid-1"],
        "preview_count": 2,
        "executor": "RDKit input validation",
    }
    assert len(state.molecules) == 2
    assert eligible_stage(state) == "generate_morgan_fingerprints"


def test_inspection_rejects_wrong_shape(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame([{"id": "one", "smiles": "CCO"}]).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="expected 3 rows"):
        inspect_library(WorkflowState(), sample, expected_rows=3)
```

- [ ] **Step 2: Run the tests and confirm the intended failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py -v
```

Expected: collection fails because `chemistry_workflow` does not exist.

- [ ] **Step 3: Implement the minimal state and result types**

Add these exact public contracts to `chemistry_workflow.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class WorkflowPhase(StrEnum):
    NEW = "new"
    INSPECTED = "inspected"
    FINGERPRINTED = "fingerprinted"
    COMPARED = "compared"
    CLUSTERED = "clustered"
    EMBEDDED = "embedded"
    OPTIMIZED = "optimized"


@dataclass(frozen=True)
class StageResult:
    stage: str
    display_label: str
    summary: dict[str, Any]
    figures: tuple[Any, ...] = ()


@dataclass
class WorkflowState:
    phase: WorkflowPhase = WorkflowPhase.NEW
    records: list[dict[str, Any]] = field(default_factory=list)
    molecules: list[Any] = field(default_factory=list)
    fingerprints: Any = None
    similarity: Any = None
    clusters: list[list[int]] = field(default_factory=list)
    representative_records: list[dict[str, Any]] = field(default_factory=list)
    conformer_molecules: list[Any] = field(default_factory=list)
    optimization_result: Any = None
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)


_NEXT_STAGE = {
    WorkflowPhase.NEW: "inspect_library",
    WorkflowPhase.INSPECTED: "generate_morgan_fingerprints",
    WorkflowPhase.FINGERPRINTED: "measure_tanimoto_similarity",
    WorkflowPhase.COMPARED: "discover_fused_butina_clusters",
    WorkflowPhase.CLUSTERED: "embed_representative_conformers",
    WorkflowPhase.EMBEDDED: "optimize_conformers_mmff94",
    WorkflowPhase.OPTIMIZED: "submit_synthesis",
}


def eligible_stage(state: WorkflowState) -> str:
    return _NEXT_STAGE[state.phase]
```

Implement `inspect_library(state, data_path, expected_rows=256)` by moving the fixed-path parsing behavior from notebook cell `guided-12`. It must require `id` and `smiles`, preserve source row indices, parse with `Chem.MolFromSmiles`, store only valid records/molecules, report every invalid ID, cap the preview at 24, update `state.summaries`, and transition to `INSPECTED` only after validation succeeds.

- [ ] **Step 4: Run the focused tests**

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add chemistry_workflow.py tests/test_chemistry_workflow.py
git commit -m "refactor: establish deterministic chemistry workflow state"
```

### Task 2: Extract the GPU Similarity Chain

**Files:**
- Modify: `chemistry_workflow.py`
- Modify: `tests/test_chemistry_workflow.py`

- [ ] **Step 1: Add failing tests for phase order and JSON-safe summaries**

Add tests using monkeypatched nvMolKit entry points. Each fake must record its arguments and return a small tensor-like object. Cover:

```python
def test_similarity_chain_requires_phase_order(inspected_state, fake_gpu):
    fingerprint = generate_morgan_fingerprints(
        inspected_state, fingerprint_radius=3, fingerprint_size=2048
    )
    assert fingerprint.summary["entry_point"] == "MorganFingerprintGenerator"
    assert fingerprint.summary["fingerprint_radius"] == 3
    assert fingerprint.summary["fingerprint_size"] == 2048
    assert inspected_state.phase is WorkflowPhase.FINGERPRINTED

    similarity = measure_tanimoto_similarity(inspected_state)
    assert similarity.summary["entry_point"] == "crossTanimotoSimilarity"
    assert similarity.summary["matrix_shape"] == [3, 3]
    assert inspected_state.phase is WorkflowPhase.COMPARED

    clusters = discover_fused_butina_clusters(inspected_state, cluster_cutoff=0.47)
    assert clusters.summary["entry_point"] == "fused_butina"
    assert clusters.summary["cluster_cutoff"] == 0.47
    assert sorted(index for cluster in inspected_state.clusters for index in cluster) == [0, 1, 2]
    eligibility = clusters.summary["representative_eligibility"]
    assert eligibility["eligible_cluster_count"] == 3
    assert eligibility["eligible_singleton_count"] == 1
    assert eligibility["maximum_representative_count"] == 3
    assert eligibility["candidates_by_cluster"] == [
        {
            "cluster_id": 0,
            "candidate_ids": ["mol-0"],
            "source_rows": [0],
            "is_singleton": True,
        },
        {
            "cluster_id": 1,
            "candidate_ids": ["mol-1"],
            "source_rows": [1],
            "is_singleton": True,
        },
        {
            "cluster_id": 2,
            "candidate_ids": ["mol-2"],
            "source_rows": [2],
            "is_singleton": True,
        },
    ]
    assert inspected_state.phase is WorkflowPhase.CLUSTERED


def test_similarity_chain_rejects_out_of_phase_calls(inspected_state):
    with pytest.raises(RuntimeError, match="fingerprinted state"):
        measure_tanimoto_similarity(inspected_state)


@pytest.mark.parametrize("cutoff", [0.399, 0.601])
def test_cluster_cutoff_is_bounded(compared_state, cutoff):
    with pytest.raises(ValueError, match="0.40 through 0.60"):
        discover_fused_butina_clusters(compared_state, cluster_cutoff=cutoff)
```

- [ ] **Step 2: Confirm the tests fail for missing functions**

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py -k "similarity or cluster" -v
```

Expected: failures name the three unimplemented domain functions.

- [ ] **Step 3: Move the existing tested scientific behavior into the domain module**

Implement these signatures:

```python
def generate_morgan_fingerprints(
    state: WorkflowState,
    *,
    fingerprint_radius: int,
    fingerprint_size: int,
) -> StageResult:
    """Run nvMolKit Morgan fingerprints after strict parameter validation."""


def measure_tanimoto_similarity(state: WorkflowState) -> StageResult:
    """Run nvMolKit all-pairs Tanimoto and summarize the off-diagonal matrix."""


def discover_fused_butina_clusters(
    state: WorkflowState,
    *,
    cluster_cutoff: float,
) -> StageResult:
    """Run nvMolKit fused Butina and validate one assignment per molecule."""
```

Move the computation and plot behavior from notebook cells `guided-19`, `guided-25`, and `guided-31` without changing scientific formulas. Preserve GPU synchronization, packed-fingerprint shape checks, off-diagonal statistics, square/symmetric/finite/range checks, most-similar-pair handling, complete cluster assignment, singleton reporting, the Tanimoto heatmap, and cluster-size plot.

Before returning the clustering summary, use RDKit to evaluate MMFF94 eligibility for every clustered molecule. Add a JSON-safe `representative_eligibility` object containing eligible cluster count, eligible singleton count, maximum feasible representative count, and eligible candidate IDs/source rows by cluster. This evidence must exist before Nemotron chooses representative count, policy, or conformer count. Tensors, matrices, molecules, and MMFF properties remain in `WorkflowState`.

- [ ] **Step 4: Run the domain tests and unchanged GPU acceptance in skip mode**

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py tests/test_gpu_acceptance.py -v
```

Expected: domain tests pass and the GPU acceptance test is skipped without `RUN_GPU_TESTS=1`.

- [ ] **Step 5: Commit Task 2**

```bash
git add chemistry_workflow.py tests/test_chemistry_workflow.py
git commit -m "refactor: extract nvMolKit similarity workflow"
```

### Task 3: Extract Representative Conformers, Figures, and Canonical Evidence

**Files:**
- Modify: `chemistry_workflow.py`
- Modify: `tests/test_chemistry_workflow.py`
- Modify: `tests/test_gpu_acceptance.py`

- [ ] **Step 1: Add failing representative and report tests**

Define the representative and evidence contracts in tests:

```python
from chemistry_workflow import (
    EvidenceRecord,
    RepresentativePolicy,
    build_workflow_report,
    select_representatives,
)


def test_singleton_policy_never_selects_one_cluster_twice(clustered_state):
    selected, shortfall = select_representatives(
        clustered_state,
        requested_count=4,
        policy=RepresentativePolicy.INCLUDE_SINGLETON_IF_AVAILABLE,
    )
    assert len({item["cluster_id"] for item in selected}) == len(selected)
    assert shortfall == max(0, 4 - len(selected))


def test_representative_order_uses_size_then_source_row(clustered_state):
    # Fixture cluster order is deliberately not source-row sorted:
    # cluster 0: size 3, minimum source row 2
    # cluster 1: size 3, minimum source row 5
    # cluster 2: size 1, minimum source row 1
    selected, _shortfall = select_representatives(
        clustered_state,
        requested_count=3,
        policy=RepresentativePolicy.LARGEST_CLUSTERS_FIRST,
    )
    assert [(item["cluster_id"], item["source_row"]) for item in selected] == [
        (0, 2),
        (1, 5),
        (2, 1),
    ]


def test_report_contains_six_canonical_evidence_groups(optimized_state):
    report = build_workflow_report(optimized_state)
    assert [item.key for item in report.evidence] == [
        "E01", "E02", "E03", "E04", "E05", "E06"
    ]
    assert all(isinstance(item, EvidenceRecord) for item in report.evidence)
    with pytest.raises(FrozenInstanceError):
        report.evidence[0].payload_json = "{}"
    assert json.loads(report.evidence[0].payload_json)["raw_count"] > 0
```

Add explicit negative tests for fewer than three eligible clusters, duplicate/unknown representative provenance, incomplete conformer pairs, non-finite energies, and unreconciled convergence totals.

- [ ] **Step 2: Run the new tests and confirm missing-contract failures**

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py -k "representative or report or conformer" -v
```

Expected: failures name the missing representative, embed, optimize, and report contracts.

- [ ] **Step 3: Implement representative and evidence types**

Add these exact types:

```python
class RepresentativePolicy(StrEnum):
    LARGEST_CLUSTERS_FIRST = "largest_clusters_first"
    INCLUDE_SINGLETON_IF_AVAILABLE = "include_singleton_if_available"


@dataclass(frozen=True)
class EvidenceRecord:
    key: str
    label: str
    payload_json: str
    provenance: str


@dataclass(frozen=True)
class WorkflowReport:
    evidence: tuple[EvidenceRecord, ...]
```

Construct every `payload_json` with `json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)`. The report contains only the tuple of frozen records; it does not duplicate mutable summary dictionaries. Parse a fresh copy only when rendering or creating the hosted payload.

Use these fixed payload keys and units:

| Evidence | Required payload keys |
| --- | --- |
| `E01` | `raw_count`, `valid_count`, `invalid_count`, `invalid_ids`, `preview_count`, `count_unit="rows"` |
| `E02` | `fingerprint_radius`, `fingerprint_size_bits`, `packed_shape`, `molecule_count`, `active_bits_min`, `active_bits_median`, `active_bits_max`, `executor`, `size_unit="bits"` |
| `E03` | `matrix_shape`, `q1`, `median`, `q3`, `p90`, `max_off_diagonal`, `most_similar_pair`, `similarity_unit="Tanimoto coefficient"` |
| `E04` | `cutoff`, `cluster_count`, `singleton_count`, `singleton_fraction`, `largest_cluster_sizes`, `assignment_count`, `cutoff_unit="Tanimoto distance"` |
| `E05` | `requested_representative_count`, `selected_representative_count`, `selection_shortfall`, `representative_policy`, `representatives`, `requested_conformers_per_representative`, `generated_conformer_count`, `partial_embedding_ids`, `zero_embedding_ids`, `count_unit="conformers"` |
| `E06` | `attempted_conformer_count`, `converged_conformer_count`, `unconverged_conformer_count`, `per_conformer_records`, `selected_conformer_records`, `energy_unit="kcal/mol"`, `comparison_scope="within molecule only"` |

`representatives` contains only `molecule_id`, `source_row`, and `cluster_id`. Each `per_conformer_records` item contains `molecule_id`, `cluster_id`, `conformer_index`, `energy_kcal_mol`, and `converged`. Each `selected_conformer_records` item adds `selected_conformer_id`. `EvidenceRecord.provenance` is respectively `RDKit input validation`, `MorganFingerprintGenerator`, `crossTanimotoSimilarity`, `fused_butina`, `EmbedMolecules`, and `MMFFOptimizeMoleculesConfs`.

Add one parameterized positive test that parses every `payload_json`, asserts the exact key set above, verifies each unit string, and verifies the producing provenance. Add negative tests for a missing required key, an unexpected key, a non-finite float, and a provenance mismatch. The exact frozen ledger that passes these tests is the ledger serialized for Nemotron and the renderer.

Implement `select_representatives` exactly as the approved specification states using the eligibility analysis already stored after clustering: cluster order by original size descending then minimum source row ascending; member order by source row; reserved singleton excluded from the fill pass; visible shortfall; stop below three eligible clusters. Never label a selection centroid or medoid.

- [ ] **Step 4: Extract embedding, optimization, and static rendering**

Implement:

```python
def embed_representative_conformers(
    state: WorkflowState,
    *,
    representative_count: int,
    representative_policy: RepresentativePolicy,
    conformers_per_representative: int,
) -> StageResult:
    """Select with Python/RDKit, then run nvMolKit EmbedMolecules."""


def optimize_conformers_mmff94(state: WorkflowState) -> StageResult:
    """Run nvMolKit MMFFOptimizeMoleculesConfs and reconcile all result pairs."""


def build_workflow_report(state: WorkflowState) -> WorkflowReport:
    """Validate all artifacts and produce E01 through E06 from computed results."""
```

Move the tested conformer, device-coordinate, energy/convergence, static 3D, and optional py3Dmol behavior from notebook cell `guided-37` and result cells `guided-39` through `guided-41`. Preserve the fixed ETKDG seed, `useRandomCoords=True`, maximum 500 MMFF iterations, authoritative `(mol_indices, conf_indices)` mapping, finite-value checks, within-molecule energy ranking, static energy plot, and static optimized structures. The event metadata must separate `Python/RDKit -> select and eligibility-check representatives` from `nvMolKit -> EmbedMolecules`.

- [ ] **Step 5: Run all domain tests**

Update `tests/test_gpu_acceptance.py` to construct its deterministic repeated-molecule CSV, call `inspect_library`, `generate_morgan_fingerprints`, `measure_tanimoto_similarity`, `discover_fused_butina_clusters`, `embed_representative_conformers`, `optimize_conformers_mmff94`, and `build_workflow_report`, then assert E01 through E06. Keep the existing L4/CUDA/version, matrix, assignment, coordinate, energy, and minimum-convergence assertions. This makes the live gate exercise the extracted facade while still proving all five nvMolKit entry points.

```bash
.venv/bin/python -m pytest tests/test_chemistry_workflow.py tests/test_gpu_acceptance.py -v
```

Expected: all domain tests pass without CUDA by using deterministic fakes for nvMolKit calls.

- [ ] **Step 6: Commit Task 3**

```bash
git add chemistry_workflow.py tests/test_chemistry_workflow.py tests/test_gpu_acceptance.py
git commit -m "refactor: extract conformer workflow and canonical evidence"
```

### Task 4: Replace Independent Calls with One Persistent Nemotron Conversation

**Files:**
- Modify: `demo_agent.py`
- Rewrite: `tests/test_demo_agent.py`

- [ ] **Step 1: Replace old guided-call tests with failing stateful-loop tests**

Define strict public schemas and the conversation contract:

```python
def test_plan_and_all_tool_results_share_one_message_history(fake_client, fake_executors):
    result = run_workflow(
        user_goal="Analyze the bundled molecular library.",
        api_key=VALID_API_KEY,
        display_events=False,
        client=fake_client,
        executors=fake_executors,
    )
    roles = [message["role"] for message in result.messages]
    assert roles[0:2] == ["system", "user"]
    # One plan acknowledgement plus six scientific tool results.
    assert roles.count("tool") == 7
    assert result.turn_count == 8
    assert result.report is not None

    for assistant_index, assistant_message in enumerate(result.messages):
        tool_calls = assistant_message.get("tool_calls", [])
        if assistant_message.get("role") != "assistant" or not tool_calls:
            continue
        # The terminal synthesis call is validated locally and intentionally
        # ends the transcript without another hosted request or tool acknowledgement.
        if tool_calls[0]["function"]["name"] == "submit_synthesis":
            assert assistant_index == len(result.messages) - 1
            continue
        tool_message = result.messages[assistant_index + 1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == tool_calls[0]["id"]
        decoded = json.loads(tool_message["content"])
        assert tool_message["content"] == json.dumps(
            decoded,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )


def test_cluster_schema_is_not_exposed_before_similarity(fake_session):
    assert fake_session.eligible_tool_name() == "inspect_library"
    fake_session.advance_through("measure_tanimoto_similarity")
    assert fake_session.eligible_tool_name() == "discover_fused_butina_clusters"


@pytest.mark.parametrize(
    "bad_call",
    [missing_call(), multiple_calls(), wrong_name_call(), bad_json_call(), extra_field_call()],
)
def test_invalid_calls_never_execute(bad_call, recording_executor):
    with pytest.raises(ToolCallError):
        validate_and_execute(bad_call, recording_executor)
    assert recording_executor.calls == []


def test_skill_is_grounding_not_a_tool(fake_client):
    session = start_session(VALID_API_KEY, "Analyze the library", client=fake_client)
    assert "skills/nvmolkit/SKILL.md" in session.grounding_source
    assert "read_nvmolkit_skill" not in session.tool_names
```

Retain focused secret-safe authentication, permission, request, invalid-key, empty-response, zero-retry, and JSON-serialization tests from the current file.

- [ ] **Step 2: Run the agent tests and confirm old-interface failures**

```bash
.venv/bin/python -m pytest tests/test_demo_agent.py -v
```

Expected: failures show that the stateful session, plan, eligible-tool loop, and conclusion contracts do not exist.

- [ ] **Step 3: Implement strict models and one-call validation**

Use frozen, strict Pydantic models with `extra="forbid"`:

```python
class FingerprintArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    fingerprint_radius: Literal[2, 3]
    fingerprint_size: Literal[1024, 2048]
    decision_basis: DecisionBasis


class ClusterArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    cluster_cutoff: float = Field(ge=0.40, le=0.60)
    decision_basis: DecisionBasis


class EmbedArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    representative_count: int = Field(ge=3, le=6)
    representative_policy: Literal[
        "largest_clusters_first", "include_singleton_if_available"
    ]
    conformers_per_representative: int = Field(ge=3, le=8)
    decision_basis: DecisionBasis
```

Define `DecisionBasis` with `Annotated[str, StringConstraints(strip_whitespace=True, min_length=12, max_length=240, pattern=r"^[^\r\n`]+$")]`. It is displayed but never passed to a scientific executor. Add tests rejecting blank, multiline, backtick-containing, and over-length rationales, and confirming the clustering and embed rationales are requested only after their prerequisite summaries exist. These are concise decision summaries, not hidden chain-of-thought.

Add argument-free strict models for inspection, similarity, and optimization. Add a strict plan schema containing the exact six stages in dependency order and concise non-empty rationales. Keep the current exact-call checks: one function call, type `function`, expected name, non-empty ID, JSON-object arguments, strict Pydantic validation, no dynamic callable or model path.

- [ ] **Step 4: Implement the persistent session and bounded loop**

Add:

```python
@dataclass
class AgentSession:
    messages: list[dict[str, Any]]
    state: WorkflowState
    turn_count: int = 0

    def eligible_tool_name(self) -> str:
        return eligible_stage(self.state)


@dataclass(frozen=True)
class WorkflowResult:
    messages: tuple[dict[str, Any], ...]
    report: WorkflowReport
    conclusion: Any
    turn_count: int
```

The first request forces `submit_workflow_plan`. Append its assistant tool-call message and a matching `role="tool"` acknowledgement before continuing. Each later request exposes only the current stage schema, appends the exact assistant `tool_calls` message, validates and executes once, then appends a matching `role="tool"` message whose `tool_call_id` is identical and whose JSON-string `content` is serialized with `allow_nan=False`. The terminal `submit_synthesis` call is validated locally and intentionally ends the transcript without a tool acknowledgement because no later hosted request is made.

Use the existing NVIDIA base URL, model, `enable_thinking=False`, temperature `0.2`, `stream=False`, and `max_retries=0`. Stop after exactly one plan, six stages, and one conclusion; early text, repeated stages, or more than eight hosted response turns fails closed.

- [ ] **Step 5: Run focused agent tests**

```bash
.venv/bin/python -m pytest tests/test_demo_agent.py -v
```

Expected: the persistent-history, call-validation, bounds, turn-limit, and secret-safety tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add stateful bounded Nemotron tool loop"
```

### Task 5: Add the Checked Conclusion and Public Workflow Facade

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_chemistry_workflow.py`

- [ ] **Step 1: Add failing conclusion-validation tests**

```python
def test_conclusion_requires_all_evidence_groups(complete_report):
    conclusion = valid_conclusion(complete_report)
    conclusion["sections"][0]["evidence_keys"] = []
    with pytest.raises(ToolCallError, match="evidence"):
        validate_conclusion(conclusion, complete_report)


def test_conclusion_rejects_unknown_evidence(complete_report):
    conclusion = valid_conclusion(complete_report)
    conclusion["sections"][0]["evidence_keys"] = ["E99"]
    with pytest.raises(ToolCallError, match="unknown evidence"):
        validate_conclusion(conclusion, complete_report)


def test_conclusion_rejects_evidence_linked_to_wrong_theme(complete_report):
    conclusion = valid_conclusion(complete_report)
    similarity = next(
        section for section in conclusion["sections"]
        if section["theme"] == "similarity_structure"
    )
    similarity["evidence_keys"] = ["E04"]
    with pytest.raises(ToolCallError, match="theme evidence"):
        validate_conclusion(conclusion, complete_report)


@pytest.mark.parametrize("prose", ["There were 24 molecules.", "Similarity was 0.47."])
def test_conclusion_rejects_digit_bearing_model_prose(complete_report, prose):
    conclusion = valid_conclusion(complete_report)
    conclusion["sections"][0]["prose"] = prose
    with pytest.raises(ToolCallError, match="computed quantities"):
        validate_conclusion(conclusion, complete_report)


def test_invalid_conclusion_preserves_report_and_withholds_prose(fake_client):
    with pytest.raises(ConclusionValidationError) as error:
        run_complete_fake_workflow(fake_client, conclusion=unknown_evidence_conclusion())
    assert error.value.report.evidence
    assert error.value.rejected_prose is None
```

- [ ] **Step 2: Confirm conclusion tests fail**

```bash
.venv/bin/python -m pytest tests/test_demo_agent.py -k conclusion -v
```

Expected: failures name missing conclusion schemas and validators.

- [ ] **Step 3: Implement the strict conclusion schema**

Define six fixed themes and require each exactly once:

```python
ConclusionTheme = Literal[
    "dataset_scope",
    "molecular_representation",
    "similarity_structure",
    "clustering",
    "conformational_sampling",
    "limitations_and_next_steps",
]


class ConclusionSection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    theme: ConclusionTheme
    prose: str = Field(min_length=1, max_length=1200)
    evidence_keys: list[Literal["E01", "E02", "E03", "E04", "E05", "E06"]]


class SubmitSynthesisArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    headline: str = Field(min_length=1, max_length=160)
    sections: list[ConclusionSection] = Field(min_length=6, max_length=6)
```

Use this minimum theme-to-evidence mapping:

```python
REQUIRED_THEME_EVIDENCE = {
    "dataset_scope": frozenset({"E01"}),
    "molecular_representation": frozenset({"E02"}),
    "similarity_structure": frozenset({"E03"}),
    "clustering": frozenset({"E04"}),
    "conformational_sampling": frozenset({"E05", "E06"}),
    "limitations_and_next_steps": frozenset({"E01", "E06"}),
}
```

`validate_conclusion` must reject digit characters in model-authored headline or prose, duplicate/missing themes, empty evidence lists, unknown keys, any section missing its required theme evidence, and missing global coverage of E01 through E06. Python renders `3D`, `ETKDGv3`, `MMFF94`, all exact metrics, units, and provenance from `WorkflowReport`; those strings do not come from model prose.

- [ ] **Step 4: Complete `run_workflow(...)` and event rendering**

Expose this notebook interface while retaining keyword-only test injection:

```python
def run_workflow(
    user_goal: str,
    api_key: str,
    display_events: bool = True,
    *,
    client=None,
    executors=None,
) -> WorkflowResult:
    """Run one plan, six validated stages, and one checked conclusion."""
```

Production execution must bind the repository's fixed data and skill paths internally. Test injection may replace the client and executor registry but cannot enter hosted payloads. The renderer prints three phase headings from the same call—Nemotron plan, continuous execution, checked conclusion—and displays each `StageResult`. Invalid synthesis renders the canonical evidence table and raises a secret-safe validation error without showing rejected prose.

- [ ] **Step 5: Run the agent and domain tests together**

```bash
.venv/bin/python -m pytest tests/test_demo_agent.py tests/test_chemistry_workflow.py -v
```

Expected: all tests pass with no hosted request or CUDA dependency.

- [ ] **Step 6: Commit Task 5**

```bash
git add demo_agent.py tests/test_demo_agent.py tests/test_chemistry_workflow.py
git commit -m "feat: ground Nemotron conclusion in canonical evidence"
```

### Task 6: Rewrite the Notebook as a Thin Presentation

**Files:**
- Rewrite: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`

- [ ] **Step 1: Replace repetitive notebook assertions with a failing thin-notebook contract**

Retain setup, Launchable, health-probe, credential, fixed-artifact, nbformat, secret, and skill-provenance tests. Replace six-section/cell-order assertions with:

```python
def test_notebook_is_one_thin_stateful_agent_presentation():
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    visible_code_lines = sum(len(cell.source.splitlines()) for cell in code_cells)
    code = "\n".join(cell.source for cell in code_cells)
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]

    assert len(notebook.cells) <= 8
    assert visible_code_lines <= 150
    assert code.count("run_workflow(") == 1
    assert "USER_GOAL" in code
    assert "request_brief_interpretation" not in code
    assert "read_nvmolkit_skill" not in code
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(ast.parse(code))
    )
    assert headings == [
        "# Can an AI chemistry agent use nvMolKit?",
        "## Preflight",
        "## User request",
        "## Agent run",
        "## Boundary",
    ]
```

Also assert that notebook markdown explicitly attributes SMILES parsing to RDKit, the five named computations to nvMolKit GPU entry points, the plan to a fixed dependency workflow, and exact metrics to Python's canonical evidence report.

- [ ] **Step 2: Run the notebook tests and confirm they fail against the old notebook**

```bash
.venv/bin/python -m pytest tests/test_notebook.py -k "thin_stateful or story_intro or preflight" -v
```

Expected: the current 48-cell guided notebook violates the cell count, code line count, headings, and single-entry-point contract.

- [ ] **Step 3: Rewrite the notebook to no more than eight cells**

Build the notebook with these exact presentation units:

1. Markdown: question, bounded-autonomy definition, RDKit/nvMolKit/Brev/Nemotron responsibilities, and research boundary.
2. Markdown: `## Preflight`.
3. Code: fixed project-root discovery, hidden `NVIDIA_API_KEY` prompt, compact Python/CUDA/nvMolKit checks, and `from demo_agent import run_workflow`.
4. Markdown: `## User request`.
5. Code: one `USER_GOAL` string and `display(Markdown(f"> {USER_GOAL}"))`.
6. Markdown: `## Agent run`, explaining that the next cell streams plan, execution, and conclusion.
7. Code: `result = run_workflow(USER_GOAL, api_key, display_events=True)`.
8. Markdown: `## Boundary` with the fixed scientific limitations.

Clear every execution count and output before commit. Do not embed API keys, executed hosted text, tensors, coordinates, or generated figures in source control.

- [ ] **Step 4: Update README without changing launch instructions**

Replace the six-call/brief-interpretation description with the fixed sequence and honest bounded autonomy. State that the skill grounds one conversation, RDKit performs input parsing and display preparation, nvMolKit performs the five named GPU operations, and Python renders exact evidence. Preserve the existing Brev, credential, Secure Link, local/GPU/hosted/rendered gate separation, and scientific limitations.

- [ ] **Step 5: Run notebook, agent, skill, setup, and domain tests**

```bash
.venv/bin/python -m pytest tests/test_notebook.py tests/test_demo_agent.py tests/test_chemistry_workflow.py tests/test_skill_snapshot.py -v
```

Expected: all selected tests pass; no GPU or hosted call is made.

- [ ] **Step 6: Commit Task 6**

```bash
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py README.md
git commit -m "feat: present one stateful nvMolKit agent workflow"
```

### Task 7: Verify, Review, Merge, Publish, and Qualify on Brev

**Files:**
- Modify only if a failing acceptance gate identifies a task-owned defect.

- [ ] **Step 1: Run the complete deterministic suite from the repository root**

```bash
.venv/bin/python -m pytest -q
```

Expected: all deterministic tests pass and only the explicit GPU acceptance test skips.

- [ ] **Step 2: Prove the intended change boundary**

```bash
git diff main...HEAD --name-only
git diff main...HEAD -- requirements.txt launchable data skills tests/test_skill_snapshot.py
git diff --check
git status --short --branch
```

Expected: only the planned helper, notebook, README, and test files—including the facade-driven GPU acceptance test—changed; the protected paths produce no diff; the worktree is clean.

- [ ] **Step 3: Run final fresh-agent reviews**

Dispatch separate fresh reviewers for:

1. specification compliance and scope;
2. hosted-call safety and secret handling;
3. scientific correctness and RDKit/nvMolKit attribution;
4. notebook narrative and minimality.

Resolve every blocking finding, rerun the complete suite, and commit each fix with a narrow message.

- [ ] **Step 4: Merge to local `main`**

From the standalone repository's primary checkout:

```bash
git merge --ff-only codex/stateful-nvmolkit-agent
```

Expected: local `main` fast-forwards and retains a clean status except the pre-existing untracked `.DS_Store`.

- [ ] **Step 5: Push the standalone public repository**

```bash
git push origin main
```

Expected: `origin/main` resolves to the verified local `main` commit. Do not push or publish the surrounding workbench.

- [ ] **Step 6: Back up and update the existing Brev checkout**

Confirm the approved target remains `agents-in-ls` / `nvmolkit---nemotron-notebook-ec6247` / instance `477hk9job`. Do not switch the global Brev organization and do not create, stop, reset, or delete an instance.

```bash
/opt/homebrew/bin/brev ls --org agents-in-ls
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook remote get-url origin"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook branch --show-current"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
```

Require the exact remote `https://github.com/ktretina/nvmolkit-brev-notebook`, branch `main`, and either a clean status or exactly one modified tracked notebook. Reject unrelated modifications.

Create the backup with an atomic, non-overwriting hard-link publish from a private temporary copy, then verify identical bytes:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "backup_tmp=\$(mktemp /home/ubuntu/.nvmolkit-stateful-backup.XXXXXX) && cp /home/ubuntu/nvmolkit-brev-notebook/notebooks/nvmolkit_nemotron_demo.ipynb \"\$backup_tmp\" && ln \"\$backup_tmp\" /home/ubuntu/nvmolkit_nemotron_demo.pre-stateful-agent-20260803.ipynb && rm \"\$backup_tmp\""
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "cmp -s /home/ubuntu/nvmolkit-brev-notebook/notebooks/nvmolkit_nemotron_demo.ipynb /home/ubuntu/nvmolkit_nemotron_demo.pre-stateful-agent-20260803.ipynb"
```

`ln` must fail if the fixed backup already exists; do not overwrite it. If the only tracked modification was the backed-up notebook, restore only that path, then require a clean checkout:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook restore --source=HEAD -- notebooks/nvmolkit_nemotron_demo.ipynb"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
```

Fetch, verify that `origin/main` equals the exact public commit recorded in Step 5, fast-forward, and recheck exact HEAD and clean status:

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook fetch origin main"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook rev-parse origin/main"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook merge --ff-only origin/main"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook rev-parse HEAD"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "git -C /home/ubuntu/nvmolkit-brev-notebook status --porcelain=v1"
```

Expected: the exact existing L4 instance remains running and ready, backup bytes match the pre-update notebook, and the checkout is clean at the published commit.

- [ ] **Step 7: Run deterministic and live GPU gates on Brev**

```bash
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "cd /home/ubuntu/nvmolkit-brev-notebook && /home/ubuntu/.venv/bin/python3 -m pytest -q"
/opt/homebrew/bin/brev exec nvmolkit---nemotron-notebook-ec6247 "cd /home/ubuntu/nvmolkit-brev-notebook && RUN_GPU_TESTS=1 /home/ubuntu/.venv/bin/python3 -m pytest tests/test_gpu_acceptance.py -v"
```

Expected: deterministic suite passes with the GPU test skipped in the first command; the explicit GPU test passes on the NVIDIA L4 under CPython 3.12.

- [ ] **Step 8: Perform hosted and rendered acceptance with the user's private key**

In JupyterLab, close any stale open notebook tab without saving, refresh the file browser, reopen `/home/ubuntu/nvmolkit-brev-notebook/notebooks/nvmolkit_nemotron_demo.ipynb`, restart the kernel, and run all cells. Enter the hosted Developer API key only through the hidden prompt.

Accept only if the rendered notebook shows:

- one high-level user request;
- one concise fixed-dependency Nemotron plan;
- six linked tool turns in one conversation, with RDKit inspection distinguished from five nvMolKit GPU entry points;
- model-selected radius, fingerprint size, clustering cutoff, representative policy/count, and conformer count within bounds;
- invalid-SMILES accounting and the retained compact static figures;
- E01 through E06 canonical evidence; and
- one validated conclusion with exact quantities rendered from Python evidence.

Do not claim hosted or rendered acceptance until this interactive run succeeds. Leave the instance running unless the user separately authorizes a lifecycle change; it remains billable.

## Completion Evidence

Report separately:

- final public commit and repository URL;
- deterministic test result;
- live L4 GPU test result;
- hosted conversation result;
- rendered-notebook result;
- backup path;
- exact Brev instance status; and
- any gate not executed or not passed.

Do not combine local, GPU, hosted, rendered, publication, or deployment proof into a single claim.
