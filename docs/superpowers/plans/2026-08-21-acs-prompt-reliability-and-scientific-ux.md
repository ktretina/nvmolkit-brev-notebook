# ACS Prompt Reliability and Scientific UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all four ACS OpenClaw prompts objective-led and scientifically precise, make successful answers deterministic, make Prompt 3 safe to replay, and qualify the change on the existing Brev instance before publication.

**Architecture:** `acs_workshop_runner.py` remains the scientific authority. It adds closed execution provenance, a canonical `answer_markdown`, and a validated Prompt 3 replay path before chemistry execution. The prompt page and installed `TOOLS.md` tell OpenClaw to stop at the first completed result and copy that answer exactly; a separate standard-library verifier checks each live trajectory and ZIP. OpenClaw loop detection is enabled only as defense in depth.

**Tech Stack:** Python 3.12, pytest, RDKit, nvMolKit, NumPy, Pillow, Matplotlib, Bash, OpenClaw/NemoClaw 2026.7.1, Brev CLI, Git, GitHub CLI.

---

## File map

- Modify `acs_workshop_runner.py`: execution metadata, canonical answers, and validated Prompt 3 replay.
- Modify `tests/test_acs_workshop_runner.py`: focused red/green tests for every runner contract.
- Create `scripts/verify_acs_openclaw_trajectory.py`: credential-safe live trajectory and ZIP verifier.
- Create `tests/test_verify_acs_openclaw_trajectory.py`: synthetic OpenClaw and archive fixtures plus mutation tests.
- Create `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl`: sanitized fixture with the exact message/event shape observed from the approved OpenClaw 2026.7.1 QA session.
- Create `scripts/acs_live_instance_patch.sh`: reviewed apply, rollback, and between-QA reset operations for the exact running instance.
- Create `scripts/run_acs_openclaw_live_qa.py`: exact-session four-prompt driver and narrow trajectory/artifact exporter.
- Create `tests/test_acs_live_instance_ops.py`: fake-command tests for patch safety, rollback, permissions, exact-session export, and timeout behavior.
- Modify `docs/acs-fall-2026-workshop.md`: four objective-led prompt blocks.
- Modify `tests/test_acs_fall_2026_workshop_page.py`: semantic prompt rules and new byte locks.
- Modify `launchable/acs_workspace_tools.md`: active one-call and canonical-answer guidance.
- Modify `tests/test_nemoclaw_phase_zero_setup.py`: active `TOOLS.md` contract checks.
- Modify `launchable/acs_nemoclaw_launchable_setup.sh`: enable and verify loop detection.
- Modify `tests/test_acs_nemoclaw_launchable_setup.py`: loop configuration and stale-file cleanup checks.
- Modify `launchable/ACS_LAUNCHABLE_FIELDS.md`: current four-prompt Console authoring instructions.
- Delete `launchable/acs_task_prompt.txt`: unused retired prompt file.
- Modify `tests/test_acs_console_bootstrap.py`: current authoring-sheet assertions, then immutable source pin.
- Regenerate `launchable/acs_console_bootstrap.sh`: pin the accepted source implementation commit.
- After live acceptance, modify `digital-biology-examples/acsfall26/README.md` in the separate `gh-pages` checkout so it is byte-identical to the canonical local page.

Use this Python command for local tests:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest
```

Run only one heavy test command at a time on the Mac.

## Verified implementation amendments

Live CLI inspection and independent review produced these corrections. They are authoritative over older pseudocode later in this plan:

- Prompt 2 asks how Butina groups molecules from distances **derived from GPU-computed Tanimoto similarities**. Its approved prompt SHA-256 is `5d556991910812a24bb09b23cd250fd4a7157986948082fb8cc05cb3d52c1f5e`.
- NemoClaw `v0.0.102` does not provide `nemoclaw <sandbox> config`. Runtime settings use `nemoclaw <sandbox> exec --workdir /sandbox/.openclaw/workspace -- openclaw config set/get/unset`, strict JSON, and one explicit `gateway restart --quiet`.
- OpenClaw `2026.7.1` tool calls include `partialArgs`, which must decode to the exact `arguments` object. Accepted result messages require `toolName == "exec"` and `isError is false`. Events use `schemaVersion == 1` and `traceSchema == "openclaw-trajectory"`.
- Terminal objective publication uses a private intended terminal-ZIP digest and a trusted prepare, journal, then public-commit sequence. Prompt 3 replay never creates its first trust binding from public directory/ZIP copies. Raw ZIP size is capped before reading.
- The verifier independently reconstructs every canonical lesson/objective answer, validates complete causal result schemas, parses and decodes complete PNG chunk streams, and checks both local and central ZIP headers without extraction.

### Task 1: Add closed execution provenance to compact stage results

**Files:**
- Modify: `tests/test_acs_workshop_runner.py:199-204,905-950`
- Modify: `acs_workshop_runner.py:201-265,1485-1501`

- [ ] **Step 1: Write the failing execution-metadata test**

Add this constant after `FIXED_GPU`:

```python
EXPECTED_EXECUTION = {
    "inspect_library": {
        "placement": "CPU",
        "software": "RDKit",
        "operation": "library parsing and validation",
        "upstream": None,
        "gpu": None,
    },
    "generate_morgan_fingerprints": {
        "placement": "GPU",
        "software": "nvMolKit",
        "operation": "Morgan fingerprint generation",
        "upstream": None,
        "gpu": asdict(FIXED_GPU),
    },
    "measure_tanimoto_similarity": {
        "placement": "GPU",
        "software": "nvMolKit",
        "operation": "Tanimoto similarity calculation",
        "upstream": None,
        "gpu": asdict(FIXED_GPU),
    },
    "discover_fused_butina_clusters": {
        "placement": "CPU",
        "software": "RDKit",
        "operation": "Butina clustering",
        "upstream": {
            "stage": "measure_tanimoto_similarity",
            "placement": "GPU",
            "software": "nvMolKit",
            "operation": "Tanimoto similarity calculation",
        },
        "gpu": asdict(FIXED_GPU),
    },
    "embed_representative_conformers": {
        "placement": "GPU",
        "software": "nvMolKit",
        "operation": "ETKDGv3 conformer embedding",
        "upstream": None,
        "gpu": asdict(FIXED_GPU),
    },
    "optimize_conformers_mmff94": {
        "placement": "GPU",
        "software": "nvMolKit",
        "operation": "MMFF94 conformer optimization",
        "upstream": None,
        "gpu": asdict(FIXED_GPU),
    },
}
```

Add `from dataclasses import asdict` and `import re` to the test imports. Extend `test_run_lesson_executes_one_terminal_prefix_and_returns_closed_compact_items` so every item has exactly these keys:

```python
assert set(item) == {
    "stage",
    "result",
    "execution",
    "image_paths",
    "summary_path",
    "readme_path",
    "artifact_directory",
}
assert item["execution"] == EXPECTED_EXECUTION[item["stage"]]
assert set(item["execution"]) == {
    "placement", "software", "operation", "upstream", "gpu"
}
if item["execution"]["upstream"] is not None:
    assert set(item["execution"]["upstream"]) == {
        "stage", "placement", "software", "operation"
    }
```

Add a parametrized test that runs all three lessons with the existing fake executions and asserts that the six observed stage payloads equal `EXPECTED_EXECUTION`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py::test_run_lesson_executes_one_terminal_prefix_and_returns_closed_compact_items
```

Expected: FAIL because compact stage items do not contain `execution`.

- [ ] **Step 3: Implement the closed execution mapping**

Add beside `StageSpec`:

```python
@dataclass(frozen=True)
class ExecutionSpec:
    placement: str
    software: str
    operation: str
    upstream_stage: str | None = None


EXECUTION_SPECS: Final = {
    "inspect_library": ExecutionSpec(
        "CPU", "RDKit", "library parsing and validation"
    ),
    "generate_morgan_fingerprints": ExecutionSpec(
        "GPU", "nvMolKit", "Morgan fingerprint generation"
    ),
    "measure_tanimoto_similarity": ExecutionSpec(
        "GPU", "nvMolKit", "Tanimoto similarity calculation"
    ),
    "discover_fused_butina_clusters": ExecutionSpec(
        "CPU",
        "RDKit",
        "Butina clustering",
        upstream_stage="measure_tanimoto_similarity",
    ),
    "embed_representative_conformers": ExecutionSpec(
        "GPU", "nvMolKit", "ETKDGv3 conformer embedding"
    ),
    "optimize_conformers_mmff94": ExecutionSpec(
        "GPU", "nvMolKit", "MMFF94 conformer optimization"
    ),
}

_GPU_IDENTITY_KEYS: Final = {
    "name", "device", "torch_version", "nvmolkit_version"
}
```

Add this validator before `_compact_stage_item`:

```python
def _execution_payload(
    stage_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    spec = EXECUTION_SPECS[stage_name]
    raw_gpu = summary.get("gpu")
    if raw_gpu is None:
        gpu = None
    elif (
        type(raw_gpu) is not dict
        or set(raw_gpu) != _GPU_IDENTITY_KEYS
        or any(type(value) is not str or not value for value in raw_gpu.values())
    ):
        raise RuntimeError("Workshop stage GPU identity is invalid.")
    else:
        gpu = dict(raw_gpu)

    if spec.placement == "GPU" and gpu is None:
        raise RuntimeError("Workshop stage GPU identity is invalid.")
    if spec.placement == "CPU" and spec.upstream_stage is None and gpu is not None:
        raise RuntimeError("Workshop stage GPU identity is invalid.")

    upstream = None
    if spec.upstream_stage is not None:
        source = EXECUTION_SPECS[spec.upstream_stage]
        if source.placement != "GPU" or gpu is None:
            raise RuntimeError("Workshop stage GPU provenance is invalid.")
        upstream = {
            "stage": spec.upstream_stage,
            "placement": source.placement,
            "software": source.software,
            "operation": source.operation,
        }

    return {
        "placement": spec.placement,
        "software": spec.software,
        "operation": spec.operation,
        "upstream": upstream,
        "gpu": gpu,
    }
```

Add this field to `_compact_stage_item`:

```python
"execution": _execution_payload(stage_name, summary),
```

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run the focused test from Step 2 plus the new all-stage test.

Expected: both tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
git commit -m "feat: report ACS stage execution provenance"
```

### Task 2: Add canonical lesson answers

**Files:**
- Modify: `tests/test_acs_workshop_runner.py:905-1290`
- Modify: `acs_workshop_runner.py:1485-1545,2555-2621`

- [ ] **Step 1: Write failing canonical-answer tests**

Add:

```python
@pytest.mark.parametrize(
    ("lesson", "terminal_stage"),
    tuple(runner.LESSON_TERMINAL_STAGES.items()),
)
def test_lesson_answers_have_exact_closed_markdown_contract(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    lesson: str,
    terminal_stage: str,
) -> None:
    execution = (
        _objective_execution(workflow_executions)
        if lesson == "sampled-3d-geometry"
        else workflow_executions[terminal_stage]
    )
    result = runner.run_lesson(
        lesson,
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    answer = result["answer_markdown"]
    assert [line for line in answer.splitlines() if line.startswith("## ")] == [
        "## Question",
        "## What ran",
        "## Measured result",
        "## Meaning",
        "## Scientific limit",
        "## Image and download location",
    ]
    assert answer.count("\n- ") in {1, 2, 3}
    assert "**Download Results**" in answer
    assert "`workshop/results.zip`" in answer
    assert answer.endswith(EXPECTED_LESSON_MEDIA[lesson])
    assert not re.search(r"\b(?:accelerat\w*|speedup|faster)\b", answer, re.I)


def test_lesson_answers_ground_cpu_gpu_work_without_performance_claims(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    result = runner.run_lesson(
        "relationships-and-groups",
        paths=workshop_paths,
        workflow_executor=lambda _stage: workflow_executions[
            "discover_fused_butina_clusters"
        ],
    )
    answer = result["answer_markdown"]
    expected_what_ran = (
        "nvMolKit generated Morgan fingerprints and computed Tanimoto "
        "similarities on GPU NVIDIA L4 (cuda:0). RDKit ran Butina clustering "
        "on CPU using those GPU-computed similarities."
    )
    assert answer == (
        "## Question\nWhich molecules are similar, and how does Butina group "
        "them from distances derived from GPU-computed Tanimoto "
        "similarities?\n\n"
        f"## What ran\n{expected_what_ran}\n\n"
        "## Measured result\n"
        f"- {EXPECTED_STAGE_RESULTS['measure_tanimoto_similarity']}\n"
        f"- {EXPECTED_STAGE_RESULTS['discover_fused_butina_clusters']}\n\n"
        "## Meaning\nThe similarity stage compares the fixed fingerprints; "
        "Butina then groups molecules whose Tanimoto distances satisfy the "
        "fixed rule.\n\n"
        "## Scientific limit\nThe cutoff 0.40 is Tanimoto distance, not "
        "similarity. Results depend on the radius-2, 1024-bit hashed "
        "fingerprint, and similarity 1.0 does not prove molecular identity or "
        "biological behavior.\n\n"
        "## Image and download location\nThe current bundle is in **Download "
        "Results** at `workshop/results.zip`.\n\n"
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "04-clusters/cluster_sizes.png"
    )
    assert "cutoff 0.40 is Tanimoto distance, not similarity" in answer
    assert "similarity 1.0 does not prove molecular identity" in answer
    assert not re.search(r"\b(?:accelerat\w*|speedup|faster)\b", answer, re.I)
```

For each lesson, the first test asserts:

```python
answer = result["answer_markdown"]
assert [line for line in answer.splitlines() if line.startswith("## ")] == [
    "## Question",
    "## What ran",
    "## Measured result",
    "## Meaning",
    "## Scientific limit",
    "## Image and download location",
]
assert answer.count("\n- ") in {1, 2, 3}
assert "**Download Results**" in answer
assert "`workshop/results.zip`" in answer
assert answer.endswith(EXPECTED_LESSON_MEDIA[lesson])
assert not re.search(r"\b(?:accelerat\w*|speedup|faster)\b", answer, re.I)
```

Use this exact media map:

```python
EXPECTED_LESSON_MEDIA = {
    "data-and-representation": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "01-inspection/library_preview.png"
    ),
    "relationships-and-groups": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "04-clusters/cluster_sizes.png"
    ),
    "sampled-3d-geometry": (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
        "06-mmff94/optimized_structures.png"
    ),
}
```

- [ ] **Step 2: Run the answer tests and confirm RED**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py -k 'lesson_answers'
```

Expected: FAIL with missing `answer_markdown`.

- [ ] **Step 3: Implement one canonical Markdown constructor**

Add:

```python
@dataclass(frozen=True)
class LessonAnswerSpec:
    question: str
    meaning: str
    scientific_limit: str
    media_line: str


LESSON_ANSWER_SPECS: Final = {
    "data-and-representation": LessonAnswerSpec(
        question=(
            "What is in the fixed molecule library, and how is it represented "
            "for comparison?"
        ),
        meaning=(
            "The validated molecules were converted into fixed-length structural "
            "descriptors that support comparisons within this exercise."
        ),
        scientific_limit=(
            "This is a deterministic 256-record ChEMBL convenience sample, not "
            "representative chemical space. Morgan and Tanimoto conclusions depend "
            "on the radius-2, 1024-bit hashed fingerprint."
        ),
        media_line=(
            "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
            "01-inspection/library_preview.png"
        ),
    ),
    "relationships-and-groups": LessonAnswerSpec(
        question=(
            "Which molecules are similar, and how does Butina group them from "
            "distances derived from GPU-computed Tanimoto similarities?"
        ),
        meaning=(
            "The similarity stage compares the fixed fingerprints; Butina then "
            "groups molecules whose Tanimoto distances satisfy the fixed rule."
        ),
        scientific_limit=(
            "The cutoff 0.40 is Tanimoto distance, not similarity. Results depend "
            "on the radius-2, 1024-bit hashed fingerprint, and similarity 1.0 does "
            "not prove molecular identity or biological behavior."
        ),
        media_line=(
            "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
            "04-clusters/cluster_sizes.png"
        ),
    ),
    "sampled-3d-geometry": LessonAnswerSpec(
        question="What sampled 3D geometries were generated and optimized?",
        meaning=(
            "The fixed representatives received deterministic ETKDGv3 conformer "
            "samples followed by within-molecule MMFF94 optimization."
        ),
        scientific_limit=(
            "The selected molecules are not centroids, medoids, or globally optimal "
            "representatives. Sampled conformers are not experimental structures, "
            "and MMFF94 energies compare sampled conformers within one molecule only."
        ),
        media_line=(
            "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
            "06-mmff94/optimized_structures.png"
        ),
    ),
}
```

Add:

```python
def _answer_markdown(
    *,
    question: str,
    what_ran: str,
    measured_results: Sequence[str],
    meaning: str,
    scientific_limit: str,
    media_line: str,
) -> str:
    if not 1 <= len(measured_results) <= 3:
        raise RuntimeError("Workshop answer facts are invalid.")
    measured = "\n".join(f"- {item}" for item in measured_results)
    return (
        f"## Question\n{question}\n\n"
        f"## What ran\n{what_ran}\n\n"
        f"## Measured result\n{measured}\n\n"
        f"## Meaning\n{meaning}\n\n"
        f"## Scientific limit\n{scientific_limit}\n\n"
        "## Image and download location\n"
        "The current bundle is in **Download Results** at "
        "`workshop/results.zip`.\n\n"
        f"{media_line}"
    )


def _execution_sentence(item: dict[str, Any]) -> str:
    execution = _fact_dict(item, "execution")
    gpu = execution["gpu"]
    upstream = execution["upstream"]
    if execution["placement"] == "GPU":
        return (
            f'{execution["software"]} ran {execution["operation"]} on GPU '
            f'{gpu["name"]} ({gpu["device"]}).'
        )
    if upstream is None:
        return f'{execution["software"]} ran {execution["operation"]} on CPU.'
    return (
        f'{execution["software"]} ran {execution["operation"]} on CPU using '
        f'{upstream["software"]} {upstream["operation"]} results computed on '
        f'GPU {gpu["name"]} ({gpu["device"]}).'
    )


def _lesson_answer_markdown(
    lesson: str,
    completed_stages: Sequence[dict[str, Any]],
) -> str:
    if tuple(item.get("stage") for item in completed_stages) != LESSON_STAGES[lesson]:
        raise RuntimeError("Workshop lesson result is invalid.")
    spec = LESSON_ANSWER_SPECS[lesson]
    what_ran = " ".join(_execution_sentence(item) for item in completed_stages)
    if lesson == "relationships-and-groups":
        gpu = _fact_dict(_fact_dict(completed_stages[0], "execution"), "gpu")
        what_ran = (
            "nvMolKit generated Morgan fingerprints and computed Tanimoto "
            f"similarities on GPU {gpu['name']} ({gpu['device']}). RDKit ran "
            "Butina clustering on CPU using those GPU-computed similarities."
        )
    return _answer_markdown(
        question=spec.question,
        what_ran=what_ran,
        measured_results=tuple(
            _fact_string(item.get("result")) for item in completed_stages
        ),
        meaning=spec.meaning,
        scientific_limit=spec.scientific_limit,
        media_line=spec.media_line,
    )


def _lesson_envelope(
    lesson: str,
    completed_stages: Sequence[dict[str, Any]],
    archive_path: Path,
) -> dict[str, Any]:
    stages = list(completed_stages)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "lesson": lesson,
        "completed_stages": stages,
        "results_zip_path": str(archive_path.resolve()),
        "artifact_relative_zip_path": "workshop/results.zip",
        "answer_markdown": _lesson_answer_markdown(lesson, stages),
    }
```

Replace the final literal return in `run_lesson` with
`_lesson_envelope(lesson, completed_stages, archive_path)`.

- [ ] **Step 4: Run the answer tests and confirm GREEN**

Expected: the canonical answer tests and existing compact-envelope tests PASS after their exact expected key sets include `answer_markdown`.

- [ ] **Step 5: Commit Task 2**

```bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
git commit -m "feat: return canonical ACS lesson answers"
```

### Task 3: Add the canonical terminal objective answer

**Files:**
- Modify: `tests/test_acs_workshop_runner.py:2034-2260`
- Modify: `acs_workshop_runner.py:1987-2051`

- [ ] **Step 1: Write failing pending and terminal answer tests**

Extend `test_third_lesson_initializes_private_objective_and_start_is_pending` with:

```python
assert "answer_markdown" not in pending
```

Add `test_terminal_objective_answer_is_canonical_and_score_bounded`. After reaching a terminal result, assert the six headings in order, exact final media line, and:

```python
answer = terminal["answer_markdown"]
baseline = terminal["baseline"]["score"]
final = terminal["final"]["score"]
assert f"Baseline `D_min`: {baseline:.3f}." in answer
assert f"Final `D_min`: {final:.3f}." in answer
assert f"Change in `D_min`: {final - baseline:+.3f}." in answer
assert "minimum pairwise Tanimoto distance" in answer
assert "min(1 - Tanimoto similarity)" in answer
assert "weakest-link diversity score within eight fixed candidates" in answer
assert "similarity score" not in answer
assert "target" not in answer.lower()
assert "predicted" not in answer.lower()
assert not re.search(r"\b(?:accelerat\w*|speedup|faster)\b", answer, re.I)
assert answer.endswith(
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "07-objective/final_panel.png"
)
```

Also assert that `What ran` contains the baseline panel IDs, the baseline limiting pair, every accepted `swap_id`, and the final panel IDs, but no intermediate score.

- [ ] **Step 2: Run the objective answer tests and confirm RED**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py -k 'objective_answer or initializes_private_objective'
```

Expected: FAIL because terminal envelopes do not contain `answer_markdown`.

- [ ] **Step 3: Implement the terminal-only answer**

Add:

```python
def _quoted_ids(values: Sequence[str]) -> str:
    if not values or any(type(value) is not str or not value for value in values):
        raise RuntimeError("Workshop objective result is invalid.")
    return ", ".join(json.dumps(value) for value in values)


def _objective_answer_markdown(
    run: ObjectiveRun,
    final: PanelMeasurement,
) -> str:
    baseline = run.baseline
    if type(baseline) is not PanelMeasurement:
        raise RuntimeError("Workshop objective result is invalid.")
    swaps = tuple(
        attempt.selected_swap.swap_id
        for attempt in run.attempts
        if attempt.selected_swap is not None
    )
    swap_text = _quoted_ids(swaps) if swaps else "none"
    limiting_pair = _quoted_ids(baseline.limiting_pairs[0])
    return _answer_markdown(
        question=(
            "Can a bounded agent improve the weakest-link diversity of a "
            "four-molecule panel?"
        ),
        what_ran=(
            f"Baseline panel: {_quoted_ids(baseline.selected_ids)}. "
            f"Baseline limiting pair: {limiting_pair}. Accepted swaps: "
            f"{swap_text}. Final panel: {_quoted_ids(final.selected_ids)}. "
            "Python validated each displayed maximum-score action against the "
            "fixed Tanimoto distance matrix derived from nvMolKit GPU-computed "
            "Morgan fingerprints and similarities."
        ),
        measured_results=(
            f"Baseline `D_min`: {baseline.score:.3f}.",
            f"Final `D_min`: {final.score:.3f}.",
            f"Change in `D_min`: {final.score - baseline.score:+.3f}.",
        ),
        meaning=(
            "A larger `D_min` means the least separated pair in the selected "
            "panel became more separated in this fingerprint space."
        ),
        scientific_limit=(
            "`D_min` is the minimum pairwise Tanimoto distance, "
            "`min(1 - Tanimoto similarity)`, and the weakest-link diversity "
            "score within eight fixed candidates. This structural-descriptor "
            "objective does not demonstrate unrestricted autonomous design or "
            "biological performance."
        ),
        media_line=(
            "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
            "07-objective/final_panel.png"
        ),
    )
```

Build the terminal dictionary in `_terminal_envelope`, assign it to `result`, add:

```python
result["answer_markdown"] = _objective_answer_markdown(run, final)
return result
```

Do not change `_pending_envelope`.

- [ ] **Step 4: Run focused objective and history tests and confirm GREEN**

Run the tests from Step 2 plus:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py -k 'objective_step or private_json or terminal_publication'
```

Expected: all selected tests PASS, with stored terminal `last_result` including the same canonical answer.

- [ ] **Step 5: Commit Task 3**

```bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
git commit -m "feat: return canonical ACS objective answer"
```

### Task 4: Replay a validated Prompt 3 result before chemistry execution

**Files:**
- Modify: `tests/test_acs_workshop_runner.py:2034-2591`
- Modify: `acs_workshop_runner.py:2336-2484,2555-2621`

- [ ] **Step 1: Write the pending replay and first-run tests**

Add this snapshot helper in the test file:

```python
def _prompt_3_snapshot(paths: runner.WorkshopPaths) -> dict[str, tuple[bytes, int]]:
    targets = [
        paths.context_path,
        paths.history_path,
        paths.output_root / "results.zip",
    ]
    for directory_name in ("05-conformers", "06-mmff94"):
        targets.extend(sorted((paths.output_root / directory_name).iterdir()))
    return {
        str(path.relative_to(paths.root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in targets
    }
```

Add:

```python
def test_third_lesson_first_call_executes_once_and_pending_replay_is_stable(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(workflow_executions)
    calls: list[str] = []

    def first_executor(stage_name: str) -> runner.WorkflowExecution:
        calls.append(stage_name)
        return execution

    first = runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=first_executor,
    )
    before = _prompt_3_snapshot(workshop_paths)

    def unexpected_executor(_stage_name: str) -> runner.WorkflowExecution:
        raise AssertionError("validated replay must not execute chemistry")

    replay = runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=unexpected_executor,
    )

    assert calls == ["optimize_conformers_mmff94"]
    assert replay == first
    assert "replayed" not in replay
    assert _prompt_3_snapshot(workshop_paths) == before
```

- [ ] **Step 2: Write fail-closed replay mutation tests**

Add three tests. Each first creates a valid pending Prompt 3 state, passes an executor that raises if called, and then mutates only the stated boundary:

```python
def _initialize_pending_geometry(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    execution = _objective_execution(workflow_executions)
    runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )


def _unexpected_geometry_executor(_stage_name: str) -> runner.WorkflowExecution:
    raise AssertionError("validated replay must not execute chemistry")


def test_third_lesson_replay_rejects_half_present_private_state_before_executor(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    _initialize_pending_geometry(workshop_paths, workflow_executions)
    history_before = workshop_paths.history_path.read_bytes()
    workshop_paths.context_path.unlink()
    with pytest.raises(RuntimeError, match="objective state"):
        runner.run_lesson(
            "sampled-3d-geometry",
            paths=workshop_paths,
            workflow_executor=_unexpected_geometry_executor,
        )
    assert workshop_paths.history_path.read_bytes() == history_before


def test_third_lesson_replay_rejects_tampered_bound_zip_before_executor(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    _initialize_pending_geometry(workshop_paths, workflow_executions)
    state_before = (
        workshop_paths.context_path.read_bytes(),
        workshop_paths.history_path.read_bytes(),
    )
    archive = workshop_paths.output_root / "results.zip"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="objective state"):
        runner.run_lesson(
            "sampled-3d-geometry",
            paths=workshop_paths,
            workflow_executor=_unexpected_geometry_executor,
        )
    assert (
        workshop_paths.context_path.read_bytes(),
        workshop_paths.history_path.read_bytes(),
    ) == state_before


def test_third_lesson_replay_rejects_valid_but_unbound_stage_file_before_executor(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
) -> None:
    _initialize_pending_geometry(workshop_paths, workflow_executions)
    image_path = workshop_paths.output_root / "05-conformers" / "embedding_counts.png"
    with Image.open(image_path) as source:
        changed = source.convert("RGBA")
    changed.save(image_path, format="PNG", compress_level=1)
    changed.close()
    with pytest.raises(RuntimeError, match="objective state"):
        runner.run_lesson(
            "sampled-3d-geometry",
            paths=workshop_paths,
            workflow_executor=_unexpected_geometry_executor,
        )
```

The ZIP test must also record the pre-mutation state bytes and prove the failed replay does not rewrite them.

Add the terminal-state replay test before implementation:

```python
def test_third_lesson_terminal_replay_skips_executor_and_is_byte_stable(
    workshop_paths: runner.WorkshopPaths,
    workflow_executions: dict[str, runner.WorkflowExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _objective_execution(
        workflow_executions, controlled_context_with_tied_paths(True)
    )
    first = runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=lambda _stage: execution,
    )
    objective = runner.objective_start(paths=workshop_paths)
    while not objective["terminal"]:
        maximum = max(action["predicted_score"] for action in objective["actions"])
        selected = next(
            action
            for action in objective["actions"]
            if action["predicted_score"] == maximum
        )
        objective = runner.objective_step(
            objective["state_id"], selected["swap_id"], paths=workshop_paths
        )
    before = _prompt_3_snapshot(workshop_paths)
    monkeypatch.setattr(runner, "_objective_render_state", pytest.fail)
    monkeypatch.setattr(runner, "objective_figures", pytest.fail)

    replay = runner.run_lesson(
        "sampled-3d-geometry",
        paths=workshop_paths,
        workflow_executor=_unexpected_geometry_executor,
    )

    assert objective["terminal"] is True
    assert replay == first
    assert replay["answer_markdown"] == first["answer_markdown"]
    assert "replayed" not in replay
    assert _prompt_3_snapshot(workshop_paths) == before
```

- [ ] **Step 3: Run the replay tests and confirm RED**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py -k 'third_lesson and replay'
```

Expected: both pending and terminal stable replay tests FAIL because the executor is still called. The terminal test must also prove that any attempted objective re-render fails.

- [ ] **Step 4: Implement bound stage reconstruction**

Add:

```python
def _bound_stage_item(
    stage_name: str,
    paths: WorkshopPaths,
    stage_members: dict[str, bytes],
) -> dict[str, Any]:
    directory = paths.output_root / STAGE_DIRECTORIES[stage_name]
    try:
        _validate_stage_directory(stage_name, directory)
        actual = _stage_directory_snapshot(stage_name, directory)
    except RuntimeError as error:
        raise _objective_error() from error
    prefix = f"{STAGE_DIRECTORIES[stage_name]}/"
    expected = {
        name.removeprefix(prefix): contents
        for name, contents in stage_members.items()
        if name.startswith(prefix)
    }
    if actual != expected:
        raise _objective_error()
    try:
        summary = json.loads(actual["summary.json"].decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise _objective_error() from error
    if (
        type(summary) is not dict
        or _formatted_json(summary).encode("utf-8") != actual["summary.json"]
    ):
        raise _objective_error()
    return _compact_stage_item(stage_name, summary, directory, STAGE_SPECS[stage_name])
```

Add the pre-execution replay function:

```python
def _validated_sampled_geometry_replay(
    paths: WorkshopPaths,
) -> dict[str, Any] | None:
    context_exists = paths.context_path.exists() or paths.context_path.is_symlink()
    history_exists = paths.history_path.exists() or paths.history_path.is_symlink()
    if not context_exists and not history_exists:
        return None
    if context_exists != history_exists:
        raise _objective_error()

    _, _, _, _, run, _ = _load_objective_state(paths)
    archive_path = paths.output_root / "results.zip"
    raw, present_stages, stage_members, objective_members = (
        _validated_results_archive(archive_path)
    )
    context_payload, _ = _read_private_json(paths.context_path)
    canonical_stage = _results_archive_bytes(present_stages, stage_members)
    binding = context_payload["stage_results_zip_sha256"]
    if hashlib.sha256(canonical_stage).hexdigest() != binding:
        raise _objective_error()
    if run is None:
        if objective_members is not None or raw != canonical_stage:
            raise _objective_error()
    else:
        objective_directory = paths.output_root / "07-objective"
        _validate_objective_directory(objective_directory, run)
        expected_objective = {
            f"07-objective/{name}": _read_regular_file(objective_directory / name)
            for name in _OBJECTIVE_FILES
        }
        expected_terminal = _results_archive_bytes(
            present_stages, stage_members, expected_objective
        )
        if objective_members != expected_objective or raw != expected_terminal:
            raise _objective_error()

    completed_stages = [
        _bound_stage_item(stage_name, paths, stage_members)
        for stage_name in LESSON_STAGES["sampled-3d-geometry"]
    ]
    return _lesson_envelope(
        "sampled-3d-geometry", completed_stages, archive_path
    )
```

In `run_lesson`, after manifest and lesson-name validation but before selecting or invoking the executor, add:

```python
if lesson == "sampled-3d-geometry":
    replay = _validated_sampled_geometry_replay(paths)
    if replay is not None:
        return replay
```

Do not catch validation errors and do not add a replay-specific response field.

- [ ] **Step 5: Run pending replay and mutation tests and confirm GREEN**

Expected: a valid replay is byte- and mtime-stable, every mutation fails before the injected executor, and a true first run executes exactly once.

- [ ] **Step 6: Run the complete replay group after implementation**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py -k 'third_lesson or objective_step_accepts_maximum'
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add acs_workshop_runner.py tests/test_acs_workshop_runner.py
git commit -m "fix: replay validated ACS geometry results"
```

### Task 5: Make the prompts and active workspace guidance objective-led

**Files:**
- Modify: `tests/test_acs_fall_2026_workshop_page.py:568-690`
- Modify: `tests/test_nemoclaw_phase_zero_setup.py:328-365`
- Modify: `docs/acs-fall-2026-workshop.md:126-246`
- Modify: `launchable/acs_workspace_tools.md:1-60`

- [ ] **Step 1: Write semantic prompt tests and confirm RED**

Add these tests to `tests/test_acs_fall_2026_workshop_page.py`:

```python
def test_prompt_blocks_lead_with_scientific_objectives_before_execution() -> None:
    for block in _prompt_blocks(_source()).values():
        positions = (
            block.index("Question:"),
            block.index("Scientific objective:"),
            block.index("Execution contract:"),
            block.index("Answer contract:"),
        )
        assert positions == tuple(sorted(positions))


def test_lesson_prompts_stop_after_the_first_complete_result() -> None:
    clauses = (
        "This command has a one-call budget.",
        "The budget is consumed when the command is submitted.",
        "top-level `status: complete`",
        "stop all tool use",
        "first completed result as authoritative",
        "Do not emit an empty response or run the command again.",
        "return its decoded `answer_markdown` string exactly",
    )
    for prompt_id, command in LESSON_COMMANDS.items():
        block = _prompt_blocks(_source())[prompt_id]
        assert block.count(command) == 1
        for clause in clauses:
            assert clause in block
        assert block.index(command) < block.index("top-level `status: complete`")
        assert block.index("top-level `status: complete`") < block.index(
            "Answer contract:"
        )


def test_prompt_science_boundaries_match_the_runner_contract() -> None:
    blocks = _prompt_blocks(_source())
    first = blocks["01-data-and-representation"]
    second = blocks["02-relationships-and-groups"]
    third = blocks["03-sampled-3d-geometry"]
    objective = blocks["04-objective"]

    assert "Do not use the words `accelerated` or `acceleration`" in first
    assert "cutoff `0.40` is Tanimoto distance, not Tanimoto similarity" in second
    assert "similarity `1.0` does not prove molecular identity" in second
    assert (
        "nvMolKit computed fingerprints and Tanimoto similarities on GPU; "
        "RDKit performed Butina clustering on CPU." in second
    )
    assert "returns both conformer stages" in third
    assert "Do not run it once per stage." in third
    assert "`D_min` is the minimum pairwise Tanimoto distance" in objective
    assert "`D_min = min(1 - Tanimoto similarity)`" in objective
    assert "Do not call `D_min` a similarity score." in objective
    assert "higher `D_min` means greater separation" in objective
    assert (
        "Do not report intermediate, predicted, target, or per-step scores "
        "anywhere in the answer." in objective
    )


def test_prompt_answer_limits_match_canonical_bullet_contract() -> None:
    blocks = _prompt_blocks(_source())
    for prompt_id in PROMPT_IDS[:3]:
        assert "Use at most three measured-result bullets." in blocks[prompt_id]
    assert (
        "Use at most three measured facts: baseline `D_min`, final `D_min`, "
        "and their change." in blocks["04-objective"]
    )
```

In the existing `test_exactly_four_self_contained_prompts_use_only_the_fixed_runner`, replace the all-prompt assertion for `at most three measured facts` with the same lesson-versus-objective split.

Run:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q tests/test_acs_fall_2026_workshop_page.py \
  -k 'scientific_objectives or first_complete or prompt_science'
```

Expected: all three new tests FAIL on missing objective-led and stop-state text.

- [ ] **Step 2: Write active `TOOLS.md` tests and confirm RED**

Add to `tests/test_nemoclaw_phase_zero_setup.py`:

```python
def test_workspace_note_stops_completed_lessons_and_copies_canonical_answers():
    source = WORKSPACE_TOOLS.read_text()
    assert "Each lesson command has a one-call budget." in source
    assert "top-level `status: complete`" in source
    assert "stop tool use and answer from that first result" in source
    assert "Never run that lesson again in the same prompt." in source
    assert "An empty assistant response does not permit another tool call." in source
    assert "Copy the decoded `answer_markdown` string exactly" in source
    assert "returns both conformer stages; run it only once" in source


def test_workspace_note_preserves_scientific_boundaries():
    source = WORKSPACE_TOOLS.read_text()
    assert "real GPU execution with no acceleration or speedup claim" in source
    assert "cutoff `0.40` is Tanimoto distance" in source
    assert "similarity `1.0` does not prove molecular identity" in source
    assert (
        "nvMolKit computes fingerprints and Tanimoto similarities on GPU; "
        "RDKit runs Butina clustering on CPU." in source
    )
    assert "`D_min` is the minimum pairwise Tanimoto distance" in source
    assert "weakest-link diversity score within eight fixed candidates" in source
    assert "Never call `D_min` a similarity score." in source
```

Run the two tests and confirm they FAIL before editing `launchable/acs_workspace_tools.md`.

- [ ] **Step 3: Rewrite the four marked prompt blocks**

Keep each marker and `~~~text` fence unchanged. Use this exact structure and wording; retain the commands and final media paths byte-for-byte.

Prompt 1:

```text
Question: What is in the fixed molecule library, and how is it represented for comparison?

Scientific objective:
Characterize the fixed molecular sample and create the structural fingerprints used by the later comparisons. Separate input validation on CPU from fingerprint generation on GPU, without making a performance claim.

Execution contract:
Work only in `/sandbox/.openclaw/workspace`.
Do not read or edit files.
Do not install software or use the network.
Do not run an alternate command.

Run only this exact command, once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson data-and-representation

This command has a one-call budget. The budget is consumed when the command is submitted.
If the command fails, report the error and stop. Do not repair or retry.
If the first returned result has top-level `status: complete`, stop all tool use. Treat that first completed result as authoritative and return its decoded `answer_markdown` string exactly. Do not emit an empty response or run the command again.

Answer contract:
The returned answer has these six headings in order: Question, What ran, Measured result, Meaning, Scientific limit, and Image and download location. Add no text before or after the decoded `answer_markdown` string. Use at most three measured-result bullets.

The scientific limits must identify the deterministic 256-record ChEMBL convenience sample as non-representative chemical space and bind conclusions to the radius-2, 1024-bit hashed fingerprint. Report real GPU execution via nvMolKit, with no CPU timing comparison or speedup claim. Do not use the words `accelerated` or `acceleration` anywhere in the answer. The download location is **Download Results** at `workshop/results.zip`.

The answer must end with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/01-inspection/library_preview.png
```

Prompt 2:

```text
Question: Which molecules are similar, and how does Butina group them from distances derived from GPU-computed Tanimoto similarities?

Scientific objective:
Find the strongest fingerprint-space relationships in the fixed sample, then group molecules with a fixed Butina distance rule. Keep similarity, distance, and CPU/GPU responsibilities distinct.

Execution contract:
Work only in `/sandbox/.openclaw/workspace`.
Do not read or edit files.
Do not install software or use the network.
Do not run an alternate command.

Run only this exact command, once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson relationships-and-groups

This command has a one-call budget. The budget is consumed when the command is submitted.
If the command fails, report the error and stop. Do not repair or retry.
If the first returned result has top-level `status: complete`, stop all tool use. Treat that first completed result as authoritative and return its decoded `answer_markdown` string exactly. Do not emit an empty response or run the command again.

Answer contract:
The returned answer has these six headings in order: Question, What ran, Measured result, Meaning, Scientific limit, and Image and download location. Add no text before or after the decoded `answer_markdown` string. Use at most three measured-result bullets.

The cutoff `0.40` is Tanimoto distance, not Tanimoto similarity. The result depends on the radius-2, 1024-bit hashed fingerprint, and similarity `1.0` does not prove molecular identity. nvMolKit computed fingerprints and Tanimoto similarities on GPU; RDKit performed Butina clustering on CPU. This execution placement is not evidence of speedup. The download location is **Download Results** at `workshop/results.zip`.

The answer must end with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png
```

Prompt 3:

```text
Question: What sampled 3D geometries were generated and optimized?

Scientific objective:
Generate a bounded set of 3D conformers for deterministic representatives and evaluate which sampled conformers converge under MMFF94. Keep sampled computational geometries separate from experimental structures.

Execution contract:
Work only in `/sandbox/.openclaw/workspace`.
Do not read or edit files.
Do not install software or use the network.
Do not run an alternate command.

Run only this exact command, once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson sampled-3d-geometry

This single command returns both conformer stages. Do not run it once per stage.
This command has a one-call budget. The budget is consumed when the command is submitted.
If the command fails, report the error and stop. Do not repair or retry.
If the first returned result has top-level `status: complete`, stop all tool use. Treat that first completed result as authoritative and return its decoded `answer_markdown` string exactly. Do not emit an empty response or run the command again.

Answer contract:
The returned answer has these six headings in order: Question, What ran, Measured result, Meaning, Scientific limit, and Image and download location. Add no text before or after the decoded `answer_markdown` string. Use at most three measured-result bullets.

The deterministic selected molecules are not centroids, medoids, or globally optimal representatives. Sampled conformers are not experimental structures. MMFF94 energies compare sampled conformers within one molecule only. Report real GPU execution without an acceleration or speedup claim. The download location is **Download Results** at `workshop/results.zip`.

The answer must end with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/06-mmff94/optimized_structures.png
```

Prompt 4:

```text
Question: Can a bounded agent improve the weakest-link diversity of a four-molecule panel?

Scientific objective:
Improve the least-separated molecular pair in a fixed four-molecule panel by choosing only state-bound swaps with the best predicted minimum distance. This is a bounded structural-diversity exercise, not open-ended molecular design.

Execution contract:
Work only in `/sandbox/.openclaw/workspace`.
Do not read or edit files.
Do not install software or use the network.
Do not run an alternate command.
Run only the exact commands below.

Run `objective-start` exactly once with this command:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-start

If `terminal` is `true`, run zero objective-step commands. If the result is pending, find the maximum numeric `predicted_score` in the displayed actions. Select one displayed action tied at that maximum; this is the best predicted `D_min`. Substitute the exact returned `state_id` and `swap_id` in this template. Keep both substituted values single-quoted; a swap ID can contain `->`.

env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-step --state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'

After each result, stop immediately when `terminal` is `true`. Otherwise, repeat with the new displayed menu. Run at most three objective-step commands in total. If a command fails, report the error and stop. Do not repair or retry.

Answer contract:
When the first terminal result arrives, stop all tool use and return its decoded `answer_markdown` string exactly. Add no text before or after it. The returned answer has these six headings in order: Question, What ran, Measured result, Meaning, Scientific limit, and Image and download location.

`D_min` is the minimum pairwise Tanimoto distance. In this exercise, `D_min = min(1 - Tanimoto similarity)`: higher `D_min` means greater separation for the least-separated pair. It is the weakest-link diversity score within eight fixed candidates. Do not call `D_min` a similarity score. Use at most three measured facts: baseline `D_min`, final `D_min`, and their change. Do not report intermediate, predicted, target, or per-step scores anywhere in the answer. Put the baseline panel, limiting pair, accepted swap or swaps, and final panel under **What ran**. This structural-descriptor objective does not demonstrate unrestricted autonomous design or biological performance. The download location is **Download Results** at `workshop/results.zip`.

The answer must end with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png
```

- [ ] **Step 4: Update active `TOOLS.md` with the same state machine**

Keep its exact four commands and media paths. Add this section after the opening paragraph:

```markdown
## Completion state

Each lesson command has a one-call budget. After the first top-level `status: complete`, stop tool use and answer from that first result. Never run that lesson again in the same prompt. An empty assistant response does not permit another tool call. Copy the decoded `answer_markdown` string exactly, with no added opening or closing text. The `sampled-3d-geometry` command returns both conformer stages; run it only once.

For the objective, run `objective-start` once, select only a displayed action tied for the maximum numeric `predicted_score`, and stop at the first terminal result. Copy its decoded `answer_markdown` string exactly.

Report real GPU execution with no acceleration or speedup claim. The cutoff `0.40` is Tanimoto distance, and similarity `1.0` does not prove molecular identity. nvMolKit computes fingerprints and Tanimoto similarities on GPU; RDKit runs Butina clustering on CPU. `D_min` is the minimum pairwise Tanimoto distance and the weakest-link diversity score within eight fixed candidates. Never call `D_min` a similarity score.
```

- [ ] **Step 5: Run semantic tests; isolate only expected byte-lock failures**

Run the complete page and phase-zero test files. Expected: all semantic tests PASS; only `test_marked_prompt_blocks_are_byte_locked` and `test_full_marked_prompt_blocks_are_byte_locked` FAIL.

- [ ] **Step 6: Replace both byte-lock dictionaries with measured hashes**

Use the existing `_prompt_blocks` and `_full_marked_prompt_blocks` helpers in this exact one-off command:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 - <<'PY'
import hashlib
import runpy

module = runpy.run_path("tests/test_acs_fall_2026_workshop_page.py")
source = module["_source"]()
for helper_name in ("_prompt_blocks", "_full_marked_prompt_blocks"):
    print(helper_name)
    for prompt_id, block in module[helper_name](source).items():
        print(prompt_id, hashlib.sha256(block.encode("utf-8")).hexdigest())
PY
```

Replace each old digest with the corresponding printed 64-character lowercase digest. Re-run the complete page and phase-zero test files; expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add docs/acs-fall-2026-workshop.md launchable/acs_workspace_tools.md \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_nemoclaw_phase_zero_setup.py
git commit -m "docs: make ACS prompts objective led"
```

### Task 6: Enable loop detection and remove retired Launchable instructions

**Files:**
- Modify: `tests/test_acs_nemoclaw_launchable_setup.py:104-219`
- Modify: `launchable/acs_nemoclaw_launchable_setup.sh:322-348,374-390`
- Modify: `tests/test_acs_console_bootstrap.py:272-285`
- Modify: `launchable/ACS_LAUNCHABLE_FIELDS.md:1-63`
- Delete: `launchable/acs_task_prompt.txt`

- [ ] **Step 1: Write the failing setup-loop test**

Add:

```python
def test_setup_enables_and_reads_back_openclaw_loop_detection() -> None:
    source = _source()
    timeout_check = '[[ "${provider_timeout}" == "300" ]] ||'
    loop_set = (
        '"${nemoclaw}" "${sandbox_name}" config set \\\n'
        "  --key tools.loopDetection.enabled \\\n"
        "  --value true \\\n"
        "  --config-accept-new-path \\\n"
        "  --restart"
    )
    loop_get = (
        'loop_detection="$("${nemoclaw}" "${sandbox_name}" config get \\\n'
        "  --key tools.loopDetection.enabled \\\n"
        '  --format json 2>/dev/null)"'
    )
    loop_check = '[[ "${loop_detection}" == "true" ]] ||'
    listener_check = 'ss -H -ltn "sport = :18789"'

    assert loop_set in source
    assert loop_get in source
    assert loop_check in source
    assert source.index(timeout_check) < source.index(loop_set)
    assert source.index(loop_set) < source.index(loop_get)
    assert source.index(loop_get) < source.index(loop_check)
    assert source.index(loop_check) < source.index(listener_check)
    assert source.count('config set \\') == 2
    assert source.count('config get \\') == 2
    assert "openclaw config set" not in source
    assert "agent --session-id" not in source
```

Run this test. Expected: FAIL because the loop setting is absent.

- [ ] **Step 2: Enable and verify loop detection**

After provider-timeout read-back validation and before `phase "Verify private dashboard"`, add:

```bash
"${nemoclaw}" "${sandbox_name}" config set \
  --key tools.loopDetection.enabled \
  --value true \
  --config-accept-new-path \
  --restart >/dev/null 2>&1
loop_detection="$("${nemoclaw}" "${sandbox_name}" config get \
  --key tools.loopDetection.enabled \
  --format json 2>/dev/null)"
[[ "${loop_detection}" == "true" ]] ||
  die "OpenClaw tool-loop detection was not enabled."
readonly loop_detection
```

Run the new test and `bash -n launchable/acs_nemoclaw_launchable_setup.sh`. Expected: PASS.

- [ ] **Step 3: Write retired-file and authoring-sheet tests**

In `tests/test_acs_nemoclaw_launchable_setup.py`, add:

```python
def test_retired_task_prompt_is_absent_but_remote_cleanup_is_retained() -> None:
    assert not (ROOT / "launchable" / "acs_task_prompt.txt").exists()
    assert _source().count("acs_task_prompt.txt") == 1
```

Replace the stale assertions in `test_authoring_sheet_uses_the_bootstrap_as_the_only_setup_source` with:

```python
assert "four fixed prompts" in source
assert "canonical `answer_markdown`" in source
assert "Paste the bootstrap" in source
assert "source push alone does not update" in source
assert "one time-bounded agent turn" not in source
assert "edit a bounded chemistry task" not in source
```

Run the two tests. Expected: FAIL while the retired source file and stale wording remain.

- [ ] **Step 4: Delete the retired source file but keep its remote cleanup entry**

Delete only `launchable/acs_task_prompt.txt` with an apply-patch deletion. Do not remove this exact setup cleanup target:

```bash
"${workspace}/acs_task_prompt.txt"
```

- [ ] **Step 5: Rewrite the Console authoring sheet**

Keep its hardware, API-key, Secure Link, access, and 16 KiB bootstrap boundaries. Replace the description with:

```markdown
- **Description:** `Use a hosted Nemotron agent for four fixed, bounded chemistry prompts: molecular validation and fingerprints, similarity and clustering, sampled 3D geometry, and a state-bound diversity objective. View the scientific images and download the validated result bundle. Computational descriptors only; not evidence of biological activity.`
```

Replace the final stale paragraph with:

```markdown
The setup installs a bounded four-prompt workshop. The runner performs deterministic chemistry, returns canonical `answer_markdown`, protects its manifest and objective state, and publishes the fixed images and ZIP. The model may call only the documented runner commands and may choose only a displayed maximum-score objective action.

A source push alone does not update the saved Launchable definition. After the source and bootstrap commits pass live acceptance, paste the exact generated `launchable/acs_console_bootstrap.sh` into Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`. Preserve the existing API-key field, one NVIDIA L4, ports `18788` and `8765`, and access settings.
```

Run the new cleanup and authoring tests. Expected: PASS.

- [ ] **Step 6: Run the complete setup and bootstrap test group**

Run:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_console_bootstrap.py
```

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add launchable/acs_nemoclaw_launchable_setup.sh \
  launchable/ACS_LAUNCHABLE_FIELDS.md \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_acs_console_bootstrap.py
git add -u launchable/acs_task_prompt.txt
git commit -m "fix: harden ACS Launchable completion controls"
```

### Task 7: Add the live OpenClaw trajectory and ZIP verifier

**Files:**
- Create: `tests/test_verify_acs_openclaw_trajectory.py`
- Create: `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl`
- Create: `scripts/verify_acs_openclaw_trajectory.py`

- [ ] **Step 1: Write a credential-free passing fixture and failing import test**

The test file loads the script as a module with `importlib.util.spec_from_file_location`. Build one synthetic OpenClaw 2026.7.1 snapshot with this exact shape:

```python
def _tool_call(call_id: str, command: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "name": "exec",
                "id": call_id,
                "arguments": {"command": command},
            }
        ],
    }


def _tool_result(call_id: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, sort_keys=True, separators=(",", ":")),
            }
        ],
    }


def _assistant_answer(answer: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }
```

Add these complete fixture helpers:

```python
def _canonical_answer(question: str, media: str, measured: tuple[str, ...]) -> str:
    return (
        f"## Question\n{question}\n\n"
        "## What ran\nThe fixed validated workflow ran.\n\n"
        "## Measured result\n"
        + "\n".join(f"- {item}" for item in measured)
        + "\n\n## Meaning\nThe result describes this fixed fingerprint-space exercise.\n\n"
        "## Scientific limit\nComputational descriptors do not establish biological "
        "activity.\n\n"
        "## Image and download location\nThe current bundle is in **Download "
        "Results** at `workshop/results.zip`.\n\n"
        f"{media}"
    )


def _lesson_result(index: int) -> dict[str, object]:
    lessons = (
        "data-and-representation",
        "relationships-and-groups",
        "sampled-3d-geometry",
    )
    questions = (
        "What is in the fixed molecule library?",
        "Which molecules are similar and grouped?",
        "What sampled 3D geometries were generated?",
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "lesson": lessons[index],
        "answer_markdown": _canonical_answer(
            questions[index], verifier.PROMPT_MEDIA[index], ("Validated result.",)
        ),
    }


def _pending_objective() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pending",
        "terminal": False,
        "state_id": "state-001",
        "actions": [
            {
                "swap_id": "mol-a->mol-e",
                "replace_id": "mol-a",
                "replacement_id": "mol-e",
                "resulting_ids": ["mol-e", "mol-b", "mol-c", "mol-d"],
                "predicted_score": 0.6,
                "score_delta": 0.1,
                "limiting_pairs": [["mol-b", "mol-c"]],
                "target_status": "target_achieved",
            },
            {
                "swap_id": "mol-b->mol-f",
                "replace_id": "mol-b",
                "replacement_id": "mol-f",
                "resulting_ids": ["mol-a", "mol-f", "mol-c", "mol-d"],
                "predicted_score": 0.55,
                "score_delta": 0.05,
                "limiting_pairs": [["mol-c", "mol-d"]],
                "target_status": "below_target",
            },
        ],
    }


def _terminal_objective() -> dict[str, object]:
    answer = (
        "## Question\nCan a bounded agent improve panel diversity?\n\n"
        "## What ran\nThe fixed maximum-score swap was validated.\n\n"
        "## Measured result\n"
        "- Baseline `D_min`: 0.500.\n"
        "- Final `D_min`: 0.600.\n"
        "- Change in `D_min`: +0.100.\n\n"
        "## Meaning\nA higher value increases weakest-pair separation.\n\n"
        "## Scientific limit\n`D_min` is the minimum pairwise Tanimoto "
        "distance, `min(1 - Tanimoto similarity)`, and the weakest-link "
        "diversity score within eight fixed candidates.\n\n"
        "## Image and download location\nThe current bundle is in **Download "
        "Results** at `workshop/results.zip`.\n\n"
        f"{verifier.PROMPT_MEDIA[3]}"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "terminal": True,
        "baseline": {"score": 0.5},
        "final": {"score": 0.6},
        "target_score": 0.58,
        "attempts": [
            {
                "score": 0.6,
                "selected_swap": {"predicted_score": 0.6},
            }
        ],
        "answer_markdown": answer,
    }


def _valid_messages(prompts: tuple[str, ...]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for index in range(3):
        call_id = f"lesson-{index + 1}"
        result = _lesson_result(index)
        messages.extend(
            (
                {"role": "user", "content": prompts[index]},
                _tool_call(call_id, verifier.LESSON_COMMANDS[index]),
                _tool_result(call_id, result),
                _assistant_answer(str(result["answer_markdown"])),
            )
        )
    pending = _pending_objective()
    terminal = _terminal_objective()
    step_command = (
        f"{verifier.RUNNER_PREFIX} objective-step --state-id 'state-001' "
        "--swap-id 'mol-a->mol-e'"
    )
    messages.extend(
        (
            {"role": "user", "content": prompts[3]},
            _tool_call("objective-start", verifier.OBJECTIVE_START),
            _tool_result("objective-start", pending),
            _tool_call("objective-step-1", step_command),
            _tool_result("objective-step-1", terminal),
            _assistant_answer(str(terminal["answer_markdown"])),
        )
    )
    return messages


def _png(width: int = 1, height: int = 1) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixels = b"\x00" + b"\x00\x00\x00\xff" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(pixels * height))
        + chunk(b"IEND", b"")
    )


def _write_valid_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", strict_timestamps=True) as archive:
        for name in verifier.REQUIRED_ZIP_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            contents = _png() if name.endswith(".png") else b"validated fixture\n"
            archive.writestr(info, contents)


def _latest_snapshot(path: Path) -> list[dict[str, object]]:
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    return event["data"]["messagesSnapshot"]


def _write_snapshot(path: Path, snapshot: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"data": {"messagesSnapshot": snapshot}}) + "\n",
        encoding="utf-8",
    )


def _user_indices(snapshot: list[dict[str, object]]) -> list[int]:
    return [index for index, item in enumerate(snapshot) if item.get("role") == "user"]


def _third_user_index(snapshot: list[dict[str, object]]) -> int:
    return _user_indices(snapshot)[2]


def _next_user_index(snapshot: list[dict[str, object]], current: int) -> int:
    return next(index for index in _user_indices(snapshot) if index > current)


def _first_final_answer_index(snapshot: list[dict[str, object]]) -> int:
    return _user_indices(snapshot)[0] + 3


def _objective_step_block(snapshot: list[dict[str, object]]) -> dict[str, object]:
    for message in snapshot:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            if (
                block.get("type") == "toolCall"
                and " objective-step " in block.get("arguments", {}).get("command", "")
            ):
                return block
    raise AssertionError("objective-step fixture block is missing")


def _valid_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    page = ROOT / "docs" / "acs-fall-2026-workshop.md"
    prompts = tuple(contract[1] for contract in verifier.load_prompt_contracts(page))
    trajectory = tmp_path / "trajectory.jsonl"
    archive = tmp_path / "results.zip"
    _write_snapshot(trajectory, _valid_messages(prompts))
    _write_valid_archive(archive)
    return trajectory, archive, page
```

Refactor `_valid_messages` and `_valid_evidence` to accept `objective_step_count: int = 1`. Build a closed state chain for each count from zero through three:

- zero steps: `objective-start` returns the terminal result;
- one step: start returns pending state 1 and the one selected step returns terminal;
- two steps: start and step 1 each return a distinct pending state, then step 2 returns terminal; and
- three steps: start and steps 1-2 each return distinct pending states, then step 3 returns terminal.

Every pending state must have its own `state_id`; its maximum-score action and the next command must use that exact state and displayed `swap_id`. Do not reuse a result object or call ID. The committed sanitized OpenClaw 2026.7.1 fixture must use the observed valid two-step shape. The compact synthetic default may remain one step.

Import `json`, `hashlib`, `importlib.util`, `stat`, `struct`, `zlib`, and `zipfile`; define `ROOT` as the repository root and bind the loaded script module to `verifier`.

Before building the synthetic fixture, inspect only the key names, roles, block types, and nesting of the already approved OpenClaw 2026.7.1 QA trajectory. Do not print or commit its original answer text, session ID, URLs, tokens, or provider metadata. Create `tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl` with that exact event/message shape. Retain only the four public approved prompt strings and their fixed public runner commands; replace all other values with credential-free synthetic fixture data. The fixture must satisfy the four approved literal prompt hashes, the observed six causal call/result pairs (three lessons, objective start, and two objective steps), and exact canonical fixture answers. Add a test that loads and validates this file through `load_messages_snapshot` and `validate_trajectory`; a parser that passes only the locally invented one-event shape is not sufficient.

Extract the four exact prompt strings from `docs/acs-fall-2026-workshop.md` with the same marker and single-fence rules as the page tests. Use three successful lesson results, then one pending `objective-start`, one maximum-score `objective-step`, and one terminal result. Each successful or terminal result contains a six-heading `answer_markdown`; the assistant answer is exactly that string.

Write one JSONL event:

```python
trajectory.write_text(
    json.dumps({"data": {"messagesSnapshot": messages}}) + "\n",
    encoding="utf-8",
)
```

Create a deterministic ZIP fixture with exactly the 34 public members listed in Step 3. Every entry uses timestamp `(1980, 1, 1, 0, 0, 0)`, deflate compression, and `(stat.S_IFREG | 0o644) << 16`. Use a small valid standard-library PNG generator for all PNG members.

Add:

```python
def test_valid_trajectory_and_archive_emit_closed_receipt(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    receipt = verifier.verify_acceptance(trajectory, archive, page)
    assert receipt == {
        "schema_version": 1,
        "status": "pass",
        "prompt_count": 4,
        "exec_call_count": 5,
        "objective_step_count": 1,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_size": archive.stat().st_size,
        "required_png_count": 4,
    }


@pytest.mark.parametrize("objective_step_count", range(4))
def test_accepts_every_bounded_objective_step_count(
    tmp_path: Path, objective_step_count: int
) -> None:
    trajectory, archive, page = _valid_evidence(
        tmp_path, objective_step_count=objective_step_count
    )
    receipt = verifier.verify_acceptance(trajectory, archive, page)
    assert receipt["exec_call_count"] == 4 + objective_step_count
    assert receipt["objective_step_count"] == objective_step_count
```

Run:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q tests/test_verify_acs_openclaw_trajectory.py
```

Expected: collection FAILS because `scripts/verify_acs_openclaw_trajectory.py` does not exist.

- [ ] **Step 2: Add trajectory mutation tests**

Add tests with these exact names and single mutations:

```python
def test_rejects_changed_or_reordered_prompt(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    snapshot[0]["content"] = snapshot[0]["content"] + " changed"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="prompt_contract"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_duplicate_prompt_3_command(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    third_turn_end = _next_user_index(snapshot, _third_user_index(snapshot))
    snapshot[third_turn_end:third_turn_end] = [
        _tool_call("duplicate-p3", verifier.LESSON_COMMANDS[2]),
        _tool_result("duplicate-p3", _lesson_result(2)),
    ]
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="command_contract"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_answer_that_differs_from_answer_markdown(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    snapshot[_first_final_answer_index(snapshot)]["content"][0]["text"] += " extra"
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="answer_contract"):
        verifier.verify_acceptance(trajectory, archive, page)


def test_rejects_nonmaximum_or_wrong_state_objective_step(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    command_block = _objective_step_block(snapshot)
    command_block["arguments"]["command"] = (
        f"{verifier.RUNNER_PREFIX} objective-step --state-id 'wrong' "
        "--swap-id 'lower-score-swap'"
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="objective_contract"):
        verifier.verify_acceptance(trajectory, archive, page)
```

Add two more pinned-contract mutations:

```python
def test_rejects_page_and_trajectory_changed_together(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    changed_page = tmp_path / "changed-page.md"
    changed_page.write_bytes(page.read_bytes().replace(b"Question:", b"Question changed:", 1))
    snapshot = _latest_snapshot(trajectory)
    original_prompt = snapshot[0]["content"]
    assert isinstance(original_prompt, str)
    snapshot[0]["content"] = original_prompt.replace(
        "Question:", "Question changed:", 1
    )
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="prompt_contract"):
        verifier.verify_acceptance(trajectory, archive, changed_page)


def test_rejects_next_call_before_prior_result(tmp_path: Path) -> None:
    trajectory, archive, page = _valid_evidence(tmp_path)
    snapshot = _latest_snapshot(trajectory)
    objective_start = _next_user_index(snapshot, _third_user_index(snapshot)) + 1
    step_call = snapshot.pop(objective_start + 2)
    snapshot.insert(objective_start + 1, step_call)
    _write_snapshot(trajectory, snapshot)
    with pytest.raises(verifier.VerificationError, match="command_contract"):
        verifier.verify_acceptance(trajectory, archive, page)
```

The first mutation must fail even though its trajectory prompt is changed to match its supplied page. This proves that the verifier trusts the four approved literal hashes, not whichever page bytes the caller supplies. The second mutation proves each tool call must receive its matching result before any next call.

Also add mutations for malformed tool-result JSON, non-complete lesson status, missing terminal objective result, a tool call after terminal, an empty assistant text block after a successful result, an extra assistant prose fragment, forbidden `accelerat*` answer text, `D_min` called similarity, and a fourth Prompt 4 measured score. Each must assert one fixed safe issue code and must not compare or print the rejected text.

- [ ] **Step 3: Add archive mutation tests**

Use this exact required member tuple in both tests and implementation:

```python
REQUIRED_ZIP_MEMBERS = (
    "README.md",
    "data/sample_molecules.csv",
    "data/PROVENANCE.md",
    "01-inspection/README.md",
    "01-inspection/summary.json",
    "01-inspection/library_preview.png",
    "02-fingerprints/README.md",
    "02-fingerprints/summary.json",
    "02-fingerprints/fingerprint_density.png",
    "03-similarity/README.md",
    "03-similarity/summary.json",
    "03-similarity/similarity_heatmap.png",
    "03-similarity/top_similarity_pairs.csv",
    "03-similarity/similarity_matrix.csv",
    "04-clusters/README.md",
    "04-clusters/summary.json",
    "04-clusters/cluster_sizes.png",
    "04-clusters/cluster_assignments.csv",
    "05-conformers/README.md",
    "05-conformers/summary.json",
    "05-conformers/embedding_counts.png",
    "06-mmff94/README.md",
    "06-mmff94/summary.json",
    "06-mmff94/conformer_energies.png",
    "06-mmff94/optimized_structures.png",
    "06-mmff94/mmff94_energies.csv",
    "06-mmff94/optimized_conformers.sdf",
    "06-mmff94/workflow_evidence.json",
    "07-objective/README.md",
    "07-objective/objective_summary.json",
    "07-objective/objective_evidence.json",
    "07-objective/score_trajectory.png",
    "07-objective/final_panel.png",
    "07-objective/final_similarity_heatmap.png",
)

REQUIRED_CHAT_PNGS = (
    "01-inspection/library_preview.png",
    "04-clusters/cluster_sizes.png",
    "06-mmff94/optimized_structures.png",
    "07-objective/final_panel.png",
)
```

Add single-mutation tests for an added `../escape`, a duplicate name, a symlink-mode entry, an encrypted flag, a ZIP comment, an extra field, wrong timestamp, unsupported compression, missing member, extra member, corrupt CRC, invalid PNG signature, zero PNG width, member larger than 8 MiB, and total declared expansion larger than 32 MiB. Every case must raise `VerificationError("archive_contract")` without extraction.

- [ ] **Step 4: Implement the standard-library verifier boundary**

The script imports only:

```python
import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence
```

Define these exact constants:

```python
RUNNER_PREFIX: Final = (
    "env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 "
    "/sandbox/.openclaw/workspace/acs_workshop_runner.py"
)
LESSON_COMMANDS: Final = (
    f"{RUNNER_PREFIX} run-lesson data-and-representation",
    f"{RUNNER_PREFIX} run-lesson relationships-and-groups",
    f"{RUNNER_PREFIX} run-lesson sampled-3d-geometry",
)
OBJECTIVE_START: Final = f"{RUNNER_PREFIX} objective-start"
OBJECTIVE_STEP_RE: Final = re.compile(
    re.escape(RUNNER_PREFIX)
    + r" objective-step --state-id '([^'\n]+)' --swap-id '([^'\n]+)'\Z"
)
PROMPT_IDS: Final = (
    "01-data-and-representation",
    "02-relationships-and-groups",
    "03-sampled-3d-geometry",
    "04-objective",
)
PROMPT_SHA256: Final = (
    "39ca26c1b494dbe01bcbaabf27d72d755b444915e9ff26c874e629f09610bf22",
    "5d556991910812a24bb09b23cd250fd4a7157986948082fb8cc05cb3d52c1f5e",
    "6779b1bfbe141a72c795d5e648ad33a5e7ddd55a8bc953b0c1ae116f757be34a",
    "ec93fcfa236b6000980178626b322aeb0786a52a53a0132338784221c24550ea",
)
HEADINGS: Final = (
    "## Question",
    "## What ran",
    "## Measured result",
    "## Meaning",
    "## Scientific limit",
    "## Image and download location",
)
PROMPT_MEDIA: Final = (
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "01-inspection/library_preview.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "04-clusters/cluster_sizes.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "06-mmff94/optimized_structures.png",
    "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/"
    "07-objective/final_panel.png",
)
MAX_TRAJECTORY_BYTES: Final = 16 * 1024 * 1024
MAX_TRAJECTORY_LINES: Final = 4096
MAX_MEMBER_BYTES: Final = 8 * 1024 * 1024
MAX_EXPANDED_BYTES: Final = 32 * 1024 * 1024
```

Define one fixed-code exception:

```python
class VerificationError(RuntimeError):
    ALLOWED = {
        "invalid_evidence",
        "prompt_contract",
        "command_contract",
        "objective_contract",
        "answer_contract",
        "archive_contract",
    }

    def __init__(self, code: str) -> None:
        if code not in self.ALLOWED:
            code = "invalid_evidence"
        super().__init__(code)
        self.code = code
```

Implement these helpers and public functions:

```python
def _read_regular(path: Path, maximum: int, code: str) -> bytes:
    try:
        mode = os.lstat(path).st_mode
        size = path.stat().st_size
    except OSError as error:
        raise VerificationError(code) from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode) or not 0 <= size <= maximum:
        raise VerificationError(code)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(code) from error
    if len(raw) != size:
        raise VerificationError(code)
    return raw


def _json_object(text: str, code: str) -> dict[str, Any]:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(code)
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=closed_object)
    except (json.JSONDecodeError, TypeError) as error:
        raise VerificationError(code) from error
    if type(value) is not dict:
        raise VerificationError(code)
    return value


def load_prompt_contracts(page_path: Path) -> tuple[tuple[str, str, str], ...]:
    raw = _read_regular(page_path, 1024 * 1024, "prompt_contract")
    try:
        source = raw.decode("utf-8")
    except UnicodeError as error:
        raise VerificationError("prompt_contract") from error
    contracts: list[tuple[str, str, str]] = []
    prior_end = -1
    for prompt_id, media_line in zip(PROMPT_IDS, PROMPT_MEDIA, strict=True):
        begin = f"<!-- ACS_PROMPT:{prompt_id}:BEGIN -->"
        end = f"<!-- ACS_PROMPT:{prompt_id}:END -->"
        if source.count(begin) != 1 or source.count(end) != 1:
            raise VerificationError("prompt_contract")
        begin_index = source.index(begin)
        end_index = source.index(end)
        if begin_index <= prior_end or end_index <= begin_index:
            raise VerificationError("prompt_contract")
        region = source[begin_index + len(begin) : end_index]
        fence = "~~~text\n"
        closing_fence = "\n~~~\n"
        if region.count(fence) != 1 or region.count(closing_fence) != 1:
            raise VerificationError("prompt_contract")
        fence_start = region.index(fence) + len(fence)
        fence_end = region.index(closing_fence, fence_start)
        if (
            region[: fence_start - len(fence)].strip()
            or region[fence_end + len(closing_fence) :].strip()
        ):
            raise VerificationError("prompt_contract")
        prompt = region[fence_start:fence_end]
        if not prompt.endswith(media_line):
            raise VerificationError("prompt_contract")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest != PROMPT_SHA256[len(contracts)]:
            raise VerificationError("prompt_contract")
        contracts.append((prompt_id, prompt, digest))
        prior_end = end_index
    return tuple(contracts)


def load_messages_snapshot(path: Path) -> list[dict[str, Any]]:
    raw = _read_regular(path, MAX_TRAJECTORY_BYTES, "invalid_evidence")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise VerificationError("invalid_evidence") from error
    if not 1 <= len(lines) <= MAX_TRAJECTORY_LINES or any(not line for line in lines):
        raise VerificationError("invalid_evidence")
    latest: list[dict[str, Any]] | None = None
    for line in lines:
        event = _json_object(line, "invalid_evidence")
        data = event.get("data")
        if type(data) is not dict or "messagesSnapshot" not in data:
            continue
        snapshot = data["messagesSnapshot"]
        if (
            type(snapshot) is not list
            or not snapshot
            or any(type(message) is not dict for message in snapshot)
        ):
            raise VerificationError("invalid_evidence")
        latest = snapshot
    if latest is None:
        raise VerificationError("invalid_evidence")
    return latest


def _single_text(message: Mapping[str, Any], code: str) -> str:
    content = message.get("content")
    if type(content) is str:
        return content
    if (
        type(content) is list
        and len(content) == 1
        and type(content[0]) is dict
        and set(content[0]) == {"type", "text"}
        and content[0]["type"] == "text"
        and type(content[0]["text"]) is str
    ):
        return content[0]["text"]
    raise VerificationError(code)


@dataclass(frozen=True)
class _CallResult:
    command: str
    result: dict[str, Any]


def _turn_evidence(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[tuple[_CallResult, ...], str]:
    call_order: list[str] = []
    commands: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    text_events: list[tuple[int, str]] = []
    first_result_position = -1
    last_result_position = -1
    pending_call_id: str | None = None
    position = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            blocks = [{"type": "text", "text": content}] if type(content) is str else content
            if type(blocks) is not list:
                raise VerificationError("command_contract")
            for block in blocks:
                position += 1
                if type(block) is not dict or type(block.get("type")) is not str:
                    raise VerificationError("command_contract")
                if block["type"] == "text":
                    if set(block) != {"type", "text"} or type(block.get("text")) is not str:
                        raise VerificationError("command_contract")
                    text_events.append((position, block["text"]))
                    continue
                if (
                    block["type"] != "toolCall"
                    or set(block) != {"type", "name", "id", "arguments"}
                    or block.get("name") != "exec"
                    or type(block.get("id")) is not str
                    or not block["id"]
                    or block["id"] in commands
                    or pending_call_id is not None
                    or type(block.get("arguments")) is not dict
                    or set(block["arguments"]) != {"command"}
                    or type(block["arguments"].get("command")) is not str
                ):
                    raise VerificationError("command_contract")
                call_order.append(block["id"])
                commands[block["id"]] = block["arguments"]["command"]
                pending_call_id = block["id"]
        elif role == "toolResult":
            position += 1
            call_id = message.get("toolCallId")
            if (
                type(call_id) is not str
                or call_id not in commands
                or call_id in results
                or call_id != pending_call_id
            ):
                raise VerificationError("command_contract")
            results[call_id] = _json_object(
                _single_text(message, "command_contract"), "command_contract"
            )
            if first_result_position < 0:
                first_result_position = position
            last_result_position = position
            pending_call_id = None
        else:
            raise VerificationError("command_contract")
    if pending_call_id is not None or set(results) != set(commands):
        raise VerificationError("command_contract")
    if any(
        event_position > first_result_position and not text.strip()
        for event_position, text in text_events
    ):
        raise VerificationError("answer_contract")
    nonempty = [(event_position, text) for event_position, text in text_events if text.strip()]
    if not nonempty or any(
        event_position <= last_result_position for event_position, _ in nonempty
    ):
        raise VerificationError("answer_contract")
    calls = tuple(_CallResult(commands[call_id], results[call_id]) for call_id in call_order)
    return calls, "".join(text for _, text in nonempty)


def _finite_score(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(code)
    number = float(value)
    if not math.isfinite(number):
        raise VerificationError(code)
    return number


def _validate_answer(
    answer: str,
    result: Mapping[str, Any],
    media_line: str,
    *,
    objective: bool,
) -> None:
    canonical = result.get("answer_markdown")
    if type(canonical) is not str or answer != canonical:
        raise VerificationError("answer_contract")
    positions: list[int] = []
    for heading in HEADINGS:
        if answer.count(heading) != 1:
            raise VerificationError("answer_contract")
        positions.append(answer.index(heading))
    if positions != sorted(positions) or not answer.endswith(media_line):
        raise VerificationError("answer_contract")
    lower = answer.lower()
    if re.search(r"\b(?:accelerat\w*|speedup|faster)\b", lower):
        raise VerificationError("answer_contract")
    if "`workshop/results.zip`" not in answer:
        raise VerificationError("answer_contract")
    if not objective:
        return
    required = (
        "minimum pairwise Tanimoto distance",
        "min(1 - Tanimoto similarity)",
        "weakest-link diversity score within eight fixed candidates",
    )
    if (
        any(text not in answer for text in required)
        or "similarity score" in lower
        or any(label in lower for label in ("intermediate", "predicted", "target"))
    ):
        raise VerificationError("answer_contract")
    baseline = result.get("baseline")
    final = result.get("final")
    if type(baseline) is not dict or type(final) is not dict:
        raise VerificationError("answer_contract")
    baseline_score = _finite_score(baseline.get("score"), "answer_contract")
    final_score = _finite_score(final.get("score"), "answer_contract")
    measured = answer.split("## Measured result\n", 1)
    if len(measured) != 2:
        raise VerificationError("answer_contract")
    measured_text = measured[1].split("\n\n## Meaning", 1)
    expected = "\n".join(
        (
            f"- Baseline `D_min`: {baseline_score:.3f}.",
            f"- Final `D_min`: {final_score:.3f}.",
            f"- Change in `D_min`: {final_score - baseline_score:+.3f}.",
        )
    )
    if len(measured_text) != 2 or measured_text[0] != expected:
        raise VerificationError("answer_contract")
    allowed_scores = {
        f"{baseline_score:.3f}",
        f"{final_score:.3f}",
        f"{final_score - baseline_score:+.3f}",
    }
    forbidden_values: list[object] = [result.get("target_score")]
    attempts = result.get("attempts")
    if type(attempts) is list:
        for attempt in attempts:
            if type(attempt) is dict:
                forbidden_values.append(attempt.get("score"))
                selected_swap = attempt.get("selected_swap")
                if type(selected_swap) is dict:
                    forbidden_values.append(selected_swap.get("predicted_score"))
    for value in forbidden_values:
        try:
            token = f"{_finite_score(value, 'answer_contract'):.3f}"
        except VerificationError:
            continue
        if token not in allowed_scores and token in answer:
            raise VerificationError("answer_contract")


def validate_trajectory(
    messages: Sequence[Mapping[str, Any]],
    contracts: Sequence[tuple[str, str, str]],
) -> tuple[int, int]:
    if len(contracts) != 4:
        raise VerificationError("prompt_contract")
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if len(user_indices) != 4:
        raise VerificationError("prompt_contract")
    turns: list[Sequence[Mapping[str, Any]]] = []
    for turn_index, (message_index, contract) in enumerate(
        zip(user_indices, contracts, strict=True)
    ):
        prompt = _single_text(messages[message_index], "prompt_contract")
        if prompt != contract[1] or hashlib.sha256(prompt.encode()).hexdigest() != contract[2]:
            raise VerificationError("prompt_contract")
        end = user_indices[turn_index + 1] if turn_index + 1 < 4 else len(messages)
        turns.append(messages[message_index + 1 : end])

    exec_count = 0
    lesson_names = (
        "data-and-representation",
        "relationships-and-groups",
        "sampled-3d-geometry",
    )
    for turn_index in range(3):
        calls, answer = _turn_evidence(turns[turn_index])
        if len(calls) != 1 or calls[0].command != LESSON_COMMANDS[turn_index]:
            raise VerificationError("command_contract")
        result = calls[0].result
        if result.get("status") != "complete" or result.get("lesson") != lesson_names[turn_index]:
            raise VerificationError("command_contract")
        _validate_answer(answer, result, PROMPT_MEDIA[turn_index], objective=False)
        exec_count += 1

    calls, answer = _turn_evidence(turns[3])
    if not 1 <= len(calls) <= 4 or calls[0].command != OBJECTIVE_START:
        raise VerificationError("objective_contract")
    current = calls[0].result
    exec_count += 1
    for call in calls[1:]:
        if current.get("terminal") is not False or current.get("status") != "pending":
            raise VerificationError("objective_contract")
        state_id = current.get("state_id")
        actions = current.get("actions")
        match = OBJECTIVE_STEP_RE.fullmatch(call.command)
        if type(state_id) is not str or not state_id or type(actions) is not list or match is None:
            raise VerificationError("objective_contract")
        command_state, swap_id = match.groups()
        if command_state != state_id or not 1 <= len(actions) <= 3:
            raise VerificationError("objective_contract")
        scores: list[tuple[str, float]] = []
        action_keys = {
            "swap_id", "replace_id", "replacement_id", "resulting_ids",
            "predicted_score", "score_delta", "limiting_pairs", "target_status",
        }
        for action in actions:
            if type(action) is not dict or set(action) != action_keys or type(action.get("swap_id")) is not str:
                raise VerificationError("objective_contract")
            scores.append(
                (
                    action["swap_id"],
                    _finite_score(action.get("predicted_score"), "objective_contract"),
                )
            )
        selected = [score for observed_id, score in scores if observed_id == swap_id]
        if len(selected) != 1 or selected[0] != max(score for _, score in scores):
            raise VerificationError("objective_contract")
        current = call.result
        exec_count += 1
    if current.get("status") != "complete" or current.get("terminal") is not True:
        raise VerificationError("objective_contract")
    _validate_answer(answer, current, PROMPT_MEDIA[3], objective=True)
    return exec_count, len(calls) - 1


def _validate_png(contents: bytes) -> None:
    if (
        len(contents) < 33
        or contents[:8] != b"\x89PNG\r\n\x1a\n"
        or struct.unpack(">I", contents[8:12])[0] != 13
        or contents[12:16] != b"IHDR"
    ):
        raise VerificationError("archive_contract")
    width, height = struct.unpack(">II", contents[16:24])
    if width <= 0 or height <= 0:
        raise VerificationError("archive_contract")


def validate_results_zip(path: Path) -> tuple[str, int, int]:
    raw = _read_regular(path, 64 * 1024 * 1024, "archive_contract")
    eocd = raw.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 != len(raw):
        raise VerificationError("archive_contract")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (
            archive.comment
            or len(names) != len(set(names))
            or tuple(names) != REQUIRED_ZIP_MEMBERS
        ):
            raise VerificationError("archive_contract")
        expanded = 0
        for info in infos:
            mode = info.external_attr >> 16
            if (
                info.is_dir()
                or info.flag_bits & 1
                or info.extra
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_DEFLATED
                or mode != stat.S_IFREG | 0o644
                or info.filename.startswith("/")
                or "\\" in info.filename
                or any(part in {"", ".", ".."} for part in info.filename.split("/"))
                or info.file_size > MAX_MEMBER_BYTES
            ):
                raise VerificationError("archive_contract")
            expanded += info.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise VerificationError("archive_contract")
        if archive.testzip() is not None:
            raise VerificationError("archive_contract")
        for name in REQUIRED_CHAT_PNGS:
            _validate_png(archive.read(name))
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        if isinstance(error, VerificationError):
            raise
        raise VerificationError("archive_contract") from error
    finally:
        if "archive" in locals():
            archive.close()
    return hashlib.sha256(raw).hexdigest(), len(raw), len(REQUIRED_CHAT_PNGS)


def verify_acceptance(
    trajectory_path: Path,
    results_zip_path: Path,
    page_path: Path,
) -> dict[str, int | str]:
    contracts = load_prompt_contracts(page_path)
    messages = load_messages_snapshot(trajectory_path)
    exec_count, objective_step_count = validate_trajectory(messages, contracts)
    archive_sha256, archive_size, png_count = validate_results_zip(results_zip_path)
    return {
        "schema_version": 1,
        "status": "pass",
        "prompt_count": 4,
        "exec_call_count": exec_count,
        "objective_step_count": objective_step_count,
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "required_png_count": png_count,
    }
```

`load_prompt_contracts` must reject symlinks, non-regular files, files larger than 1 MiB, duplicate or missing markers, more than one fenced block per marker, wrong order, or any prompt that does not end in its fixed media line. It hashes the exact decoded prompt string; it does not normalize whitespace.

`load_messages_snapshot` must reject symlinks, non-regular files, oversized files, too many lines, duplicate JSON keys, non-object events, an open `messagesSnapshot`, or no snapshot. It returns only the last snapshot and does not print evidence.

`validate_trajectory` must split on exactly four byte-equal user prompt messages. For each turn it accepts only `assistant` tool-call rows, their immediately following matching `toolResult` rows, optional whitespace-only assistant fragments before the first tool result, and one final nonempty assistant text. It rejects every empty or whitespace-only assistant fragment after the first successful tool result. A tool call is accepted only when its content is one closed `toolCall` object with `name == "exec"`, a nonempty ID, and arguments exactly `{"command": command}`. Its matching `toolResult` must name the same ID and contain one text block that decodes to one closed JSON object before any next tool call.

Prompts 1-3 each require their one exact command, one result with `status == "complete"`, matching `lesson`, and one final answer exactly equal to the returned `answer_markdown`. Prompt 4 requires `objective-start` once. Each pending result must have `terminal is False`, an exact nonempty `state_id`, and one to three closed actions. Parse the next command with `OBJECTIVE_STEP_RE`; require the returned state ID, a displayed swap ID, and a finite `predicted_score` equal to the maximum. Accept zero to three steps and require the final result to have `status == "complete"`, `terminal is True`, and `answer_markdown`. Reject any tool call after terminal.

For every final answer, require the six headings once and in order, the exact prompt media line, `workshop/results.zip`, and no case-insensitive `accelerat` prefix, `speedup`, or `faster`. For Prompt 4 also require `minimum pairwise Tanimoto distance`, `min(1 - Tanimoto similarity)`, and `weakest-link diversity score`; reject `similarity score`. Require the measured section to contain exactly the baseline, final, and change bullet lines computed from the terminal result.

`validate_results_zip` must open without extraction, require exactly `REQUIRED_ZIP_MEMBERS` in order, reject duplicate or unsafe names, directory entries, symlink or special modes, encryption, comments, extra fields, non-deflate compression, wrong timestamps or modes, per-member and total limits, CRC errors, and trailing bytes. For every path in `REQUIRED_CHAT_PNGS`, require the eight-byte PNG signature, an `IHDR` first chunk of length 13, and positive width and height from `struct.unpack(">II", data[16:24])`.

- [ ] **Step 5: Implement the closed CLI**

Use two required options and one repository-relative default:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--results-zip", required=True, type=Path)
    parser.add_argument(
        "--page",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "acs-fall-2026-workshop.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        receipt = verify_acceptance(
            arguments.trajectory, arguments.results_zip, arguments.page
        )
    except VerificationError as error:
        receipt = {
            "schema_version": 1,
            "status": "fail",
            "issue_code": error.code,
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile):
        receipt = {
            "schema_version": 1,
            "status": "fail",
            "issue_code": "invalid_evidence",
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add subprocess tests proving success prints only the seven-key pass receipt, failure prints only the three-key fail receipt, and neither output contains a prompt, answer, path, token, URL, command, or exception representation.

- [ ] **Step 6: Run verifier tests, Ruff, and scoped mypy**

Run one command at a time:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q tests/test_verify_acs_openclaw_trajectory.py
```

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff format --check \
  scripts/verify_acs_openclaw_trajectory.py \
  tests/test_verify_acs_openclaw_trajectory.py
```

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff check \
  scripts/verify_acs_openclaw_trajectory.py \
  tests/test_verify_acs_openclaw_trajectory.py
```

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/mypy --strict \
  scripts/verify_acs_openclaw_trajectory.py
```

Expected: all commands PASS. If the standalone mypy entry point is unavailable, run `python3 -m mypy --strict` with the same Python 3.12 interpreter; do not skip the gate silently.

- [ ] **Step 7: Commit Task 7**

```bash
git add scripts/verify_acs_openclaw_trajectory.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/fixtures/acs_openclaw_2026_7_1_trajectory.jsonl
git commit -m "test: verify ACS OpenClaw live trajectories"
```

### Task 8: Add tested live operations, then run integration and reviews

**Files:**
- Create: `scripts/acs_live_instance_patch.sh`
- Create: `scripts/run_acs_openclaw_live_qa.py`
- Create: `tests/test_acs_live_instance_ops.py`
- Verify: every file changed in Tasks 1-7
- Modify only if a review finds a valid specification or quality defect

- [ ] **Step 1: Bind the scripts to the current read-only CLI surface**

Re-run the exact-instance identity gate from Task 9 Steps 3 and 4. Inspect only `brev exec --help`, `nemoclaw --help`, the exact sandbox subcommand help, `agent --help`, `config get --help`, `config set --help`, any advertised `config unset --help`, and the documented exact-session trajectory/export help. Record only command names, option names, version strings, and whether unset is supported. Do not print configuration values, session lists, transcript content, or environment variables. Do not mutate the instance in this step.

If OpenClaw 2026.7.1 has no supported exact-session export, identify the fixed sandbox trajectory directory from current product help or code and require a single regular file whose basename is exactly `<session UUID>.trajectory.jsonl`. The production script must reject recursive search, multiple matches, symlinks, and any path outside the fixed session directory. It must never copy `sessions.json` or enumerate another session.

- [ ] **Step 2: Write failing live-operation tests**

Create fake `brev` and `nemoclaw` commands under a pytest temporary directory. Add tests for this exact patch interface:

```text
scripts/acs_live_instance_patch.sh \
  --mode apply|rollback|reset-between-qa \
  --bundle-dir ABSOLUTE_DIRECTORY \
  --state-dir ABSOLUTE_DIRECTORY \
  --sandbox acs-chemistry-agent
```

Require `set -Eeuo pipefail`, `umask 077`, a mode-`0700` state directory, and only mode-`0600` state/receipt files. Cover prior loop state as three distinct cases: absent, present false, and present true. For an absent value, rollback must use the version-gated exact unset operation; if unset is unavailable, apply must stop before its first mutation. Inject one failure after backup and one termination signal after install; both must restore byte-identical files/directories and the exact prior loop-state presence/value. Run rollback twice and require an idempotent safe receipt. Reject symlinks, special files, wrong user/sandbox, hash drift, path escape, broad globs, and any target outside the fixed allowlist.

Add tests for this QA interface:

```text
scripts/run_acs_openclaw_live_qa.py \
  --session-id UUID \
  --page ABSOLUTE_PAGE \
  --output-dir ABSOLUTE_MODE_0700_DIRECTORY \
  --sandbox acs-chemistry-agent
```

Require a fresh RFC 4122 UUID, the four literal `PROMPT_SHA256` values in order, exactly four agent submissions beginning at Prompt 1, no retry, and one exact-session trajectory plus `results.zip`. A timeout at any prompt must return exit `75`, write no accepted receipt, and never submit a later prompt. A non-timeout failure must return a different nonzero code. The driver must print only a closed receipt and must not print prompts, answers, commands, paths, URLs, session IDs, tokens, or exceptions. Test the exact sanitized OpenClaw 2026.7.1 fixture shape from Task 7 and prove a trajectory basename for any different UUID is rejected.

Run the new tests. Expected: collection or execution FAILS because the two scripts do not exist.

- [ ] **Step 3: Implement the rollback-safe patch script**

Implement `scripts/acs_live_instance_patch.sh` with the reviewed CLI syntax from Step 1. Its only mutable sandbox targets are:

```text
/sandbox/.openclaw/workspace/acs_workshop_runner.py
/sandbox/.openclaw/workspace/TOOLS.md
/sandbox/.openclaw/workspace/.acs-workshop-state/manifest.json
/sandbox/.openclaw/workspace/outputs/workshop
/sandbox/.openclaw/workspace/.acs-workshop-state/context.json
/sandbox/.openclaw/workspace/.acs-workshop-state/history.json
tools.loopDetection.enabled
```

Before mutation, record loop state as closed JSON with `presence` equal to `absent` or `present`; include `value` only when present and require it to be a JSON Boolean. Back up only the allowlisted existing regular files/directories below a unique mode-`0700` state child. Record a closed manifest of modes and hashes in a mode-`0600` file. Install staged bytes atomically, rebuild the six-file protected manifest, and reset only workshop outputs/context/history. Trap `ERR`, `INT`, and `TERM` after backup; every trap calls the same idempotent rollback function. Rollback restores absence, false, or true exactly, verifies hashes/modes, and emits only a closed receipt. `reset-between-qa` must preserve runner, `TOOLS.md`, manifest, loop state, services, and transcripts.

Do not use unresolved globs, recursive deletion, `pkill`, lifecycle commands, session commands, or secret-bearing configuration reads. Make the script pass `bash -n` and the fake-command tests.

- [ ] **Step 4: Implement the exact-session QA driver and commit**

Implement the standard-library Python driver with the syntax bound in Step 1. Extract the four prompts with the verifier's marker/fence rules and require the four pinned hashes before the first submission. Send all four in order to one new UUID. Copy only the exact trajectory file or exact supported export for that UUID, then copy the final ZIP. Resolve both sources below fixed roots, reject links/non-regular files, and write destination files mode `0600` inside the pre-existing mode-`0700` output directory. Scan copied evidence for common credential-key names and reject rather than print a match.

The driver does not resume or retry. Exit `75` means the caller must discard that session and restart a complete four-prompt sequence from Prompt 1 with a different fresh UUID. No partial sequence can become accepted evidence.

Run:

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m pytest -q tests/test_acs_live_instance_ops.py
bash -n scripts/acs_live_instance_patch.sh
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff format --check \
  scripts/run_acs_openclaw_live_qa.py tests/test_acs_live_instance_ops.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff check \
  scripts/run_acs_openclaw_live_qa.py tests/test_acs_live_instance_ops.py
/Library/Frameworks/Python.framework/Versions/3.12/bin/mypy --strict \
  scripts/run_acs_openclaw_live_qa.py
```

Expected: all PASS. Commit the two scripts and operations tests in one focused commit.

- [ ] **Step 5: Run the focused integration group**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_workshop_runner.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_acs_console_bootstrap.py
```

Expected: all selected tests PASS.

- [ ] **Step 6: Run the complete repository suite once**

Run:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q
```

Expected: the complete suite PASSes with no skipped test introduced by this change. Record the exact passed and skipped counts.

- [ ] **Step 7: Run shell, Node, size, format, and type gates**

Run one command at a time:

```bash
bash -n \
  launchable/acs_nemoclaw_launchable_setup.sh \
  scripts/acs_live_instance_patch.sh \
  launchable/acs_console_bootstrap.sh.in \
  launchable/acs_console_bootstrap.sh \
  launchable/nemoclaw_phase_zero.sh \
  launchable/start_artifact_server.sh
```

```bash
node --check launchable/openclaw_secure_link_proxy.mjs
node --test tests/openclaw_secure_link_proxy.test.mjs
```

```bash
test "$(LC_ALL=C wc -c < launchable/acs_console_bootstrap.sh)" -le 16384
```

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff format --check \
  acs_workshop_runner.py \
  scripts/verify_acs_openclaw_trajectory.py \
  scripts/run_acs_openclaw_live_qa.py \
  tests/test_acs_workshop_runner.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_acs_console_bootstrap.py
```

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/ruff check \
  acs_workshop_runner.py \
  scripts/verify_acs_openclaw_trajectory.py \
  scripts/run_acs_openclaw_live_qa.py \
  tests/test_acs_workshop_runner.py \
  tests/test_verify_acs_openclaw_trajectory.py \
  tests/test_acs_live_instance_ops.py \
  tests/test_acs_nemoclaw_launchable_setup.py \
  tests/test_nemoclaw_phase_zero_setup.py \
  tests/test_acs_fall_2026_workshop_page.py \
  tests/test_acs_console_bootstrap.py
```

```bash
env PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/mypy \
  --strict --ignore-missing-imports \
  acs_workshop_runner.py scripts/verify_acs_openclaw_trajectory.py \
  scripts/run_acs_openclaw_live_qa.py
```

Expected: every command exits zero.

- [ ] **Step 8: Run repository and secret gates**

```bash
git diff --check
gitleaks git \
  --log-opts="origin/acs-fall-2026-launchable..HEAD" \
  --no-banner --redact .
git status --short --branch
```

Expected: whitespace and Gitleaks gates PASS. Status contains no generated workshop output, private state, credentials, trajectories, or test cache.

- [ ] **Step 9: Request independent specification review**

Give a fresh reviewer:

- `docs/superpowers/specs/2026-08-21-acs-prompt-reliability-and-scientific-ux-design.md`;
- this plan;
- the complete diff from `4d47e36` to `HEAD`; and
- the exact test receipts.

Require findings first. The reviewer must check every design requirement, especially response key closure, replay-before-executor ordering, stage/ZIP binding, Prompt 4 definitions, prompt byte locks, stale-file deletion, loop read-back, verifier safety, and publication order. Repair every valid Critical or Important finding with a new failing test, minimal fix, and focused rerun.

- [ ] **Step 10: Request independent code-quality review**

Give a different fresh reviewer the same implementation commit range. Require checks for invalid trust boundaries, accidental mutation during replay, answer drift, permissive trajectory parsing, archive extraction or traversal, secret leakage, stale documentation, and unnecessary scope. Resolve all valid Critical or Important findings and rerun the affected tests.

- [ ] **Step 11: Re-run changed gates after review repairs**

If either review changed code, repeat Steps 5, 7, and 8. Commit each focused repair with a descriptive message. Record the final source implementation SHA:

```bash
git rev-parse HEAD
```

This SHA is the immutable source pin used in Task 9.

### Task 9: Repin the bootstrap and patch the approved running instance

**Files:**
- Modify: `launchable/acs_console_bootstrap.sh`
- Modify: `tests/test_acs_console_bootstrap.py:13-30`
- Create outside the repository: `/private/tmp/acs-prompt-reliability-20260821/` for reviewed staging, rollback scripts, and safe receipts
- Remote target only: organization `agents-in-ls`, instance `acs-fall-2026-gpu-chemistry-agent-lab-4e08c1` / `8id74izoa`

- [ ] **Step 1: Repin the generated bootstrap to the source implementation SHA**

Use apply-patch to replace the prior 40-character SHA in both:

```text
launchable/acs_console_bootstrap.sh
tests/test_acs_console_bootstrap.py:PINNED_SETUP_COMMIT
```

Do not change `launchable/acs_console_bootstrap.sh.in`. Verify exact generation:

```bash
SOURCE_SHA="$(git rev-parse HEAD)" \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 - <<'PY'
import os
from pathlib import Path

root = Path.cwd()
template = (root / "launchable/acs_console_bootstrap.sh.in").read_text()
expected = template.replace("@REVIEWED_PUBLIC_COMMIT_SHA@", os.environ["SOURCE_SHA"])
actual = (root / "launchable/acs_console_bootstrap.sh").read_text()
raise SystemExit(0 if actual == expected else 1)
PY
```

Run `tests/test_acs_console_bootstrap.py`, `bash -n`, the 16,384-byte gate, `git diff --check`, and staged Gitleaks.

- [ ] **Step 2: Commit the bootstrap repin separately**

```bash
git add launchable/acs_console_bootstrap.sh tests/test_acs_console_bootstrap.py
git commit -m "chore: repin ACS Launchable bootstrap"
```

Do not push either the source implementation commits or this bootstrap commit yet.

- [ ] **Step 3: Re-establish the exact Brev task contract**

Record this contract in the task commentary before any remote command:

```text
Task label: acs-prompt-reliability-20260821
Organization: agents-in-ls
Instance: acs-fall-2026-gpu-chemistry-agent-lab-4e08c1
Instance ID: 8id74izoa
Ownership: explicitly shared with the user who owns the open main chat
Authority: user-approved in-place runner, TOOLS.md, manifest, loop setting, workshop-output, and objective-state update on 2026-08-21
Remote host staging: /home/ubuntu/.local/state/codex-acs-prompt-reliability-20260821
Sandbox target: /sandbox/.openclaw/workspace
Processes: existing acs-chemistry-agent NemoClaw/OpenClaw only
Ports: preserve 18788, 18789, and 8765; create no local forward
GPU: existing NVIDIA L4; run only the bounded four-prompt QA workload
Lifecycle: no create, start, stop, reset, or delete authority
Cost: preserve the already-running instance; create no new billable resource
Preserve: agent:main:main transcript and all unrelated files, sessions, processes, and services
```

- [ ] **Step 4: Version-gate exact Brev commands and re-verify the instance**

Run:

```bash
/opt/homebrew/bin/brev --version
/opt/homebrew/bin/brev ls --help
/opt/homebrew/bin/brev exec --help
/opt/homebrew/bin/brev copy --help
/opt/homebrew/bin/brev ls --org agents-in-ls --json
```

Require the exact name/ID, `RUNNING`, `COMPLETED`, `READY`, and one NVIDIA L4. Through one read-only command on the exact instance, also inspect `nemoclaw acs-chemistry-agent config get --help`, `config set --help`, and any advertised `config unset --help`; do not invoke a configuration mutation yet. Do not run `brev set`, `brev refresh`, or any lifecycle command. If current help does not support the required exact-instance access, loop read-back, or rollback without shared-state mutation, stop before the patch.

- [ ] **Step 5: Build and review the rollback-aware patch bundle**

Read `acs_source_commit` from the generated bootstrap with the exact `sed` command used in Task 10 Step 2 and require a 40-character lowercase SHA. Create `/private/tmp/acs-prompt-reliability-20260821` with mode `0700` using `mkdir -p` followed by `chmod 700`. Require `git diff --exit-code "${acs_source_commit}" --` for the six paths below, then copy those exact working-tree bytes with `rsync` into the task staging directory. This binds staging to the immutable source commit rather than the later bootstrap commit.

```text
acs_workshop_runner.py
launchable/acs_workspace_tools.md
scripts/verify_acs_openclaw_trajectory.py
scripts/acs_live_instance_patch.sh
scripts/run_acs_openclaw_live_qa.py
docs/acs-fall-2026-workshop.md
```

Record their SHA-256 values in a mode-`0600` local manifest. Use the committed and locally tested `scripts/acs_live_instance_patch.sh`; do not create or improvise a different live mutation script after review. Its bound host/sandbox operations must obey these exact rules:

1. Fail unless the remote user is `ubuntu`, the NemoClaw sandbox is `acs-chemistry-agent`, and all staged SHA-256 values match the local manifest.
2. Read `tools.loopDetection.enabled` as JSON without printing the value; save closed `presence` plus Boolean `value` state in a mode-`0600` file inside the task-owned mode-`0700` rollback directory. Distinguish absent, present false, and present true. If absent cannot be restored with the version-gated unset operation, stop before mutation.
3. In the sandbox, create a unique mode-`0700` backup below `/tmp/acs-prompt-reliability-20260821`. Reject symlink or special-file targets. Copy the current runner, `TOOLS.md`, manifest, `outputs/workshop`, `context.json`, and `history.json` when present. Do not read or copy any session, key, token, provider, or gateway file.
4. Upload the reviewed runner and workspace note to a unique sandbox staging directory. Verify their hashes before install.
5. Install each through a same-directory temporary regular file, `chmod 0444`, and `os.replace`/`mv` into the fixed workspace path.
6. Rebuild `.acs-workshop-state/manifest.json` from exactly the six protected manifest files using canonical JSON, mode `0444`, and atomic replacement.
7. Remove only `outputs/workshop`, `.acs-workshop-state/context.json`, and `.acs-workshop-state/history.json` after the backup. Recreate `outputs/workshop` safely. Do not touch `.acs-workshop-state/manifest.json`.
8. Run runner `--help`, manifest verification, permissions, file-count, and hash checks before changing configuration.
9. Set `tools.loopDetection.enabled` to JSON `true` with one NemoClaw restart, then read it back and require the exact JSON literal `true`.
10. On any failure after backup, restore the exact backed-up files and directories atomically, restore the prior loop-detection value with one restart, verify restoration hashes, and return a safe nonzero receipt.
11. Emit only closed status, hash, count, mode, and rollback fields. Never emit file contents, prompts, answers, session IDs, URLs, commands, paths containing tokens, or configuration values.

Review the staged hashes against `acs_source_commit` before running either committed script. Use explicit paths only; no broad glob, `rm -rf` target, `pkill`, service-wide cleanup, or lifecycle operation is allowed.

- [ ] **Step 6: Apply the in-place patch and verify rollback readiness**

Copy the task bundle only to the exact instance, then run the reviewed host orchestrator with `brev exec`. Require a safe receipt with:

```json
{
  "status": "pass",
  "runner_hash": "the reviewed source runner SHA-256",
  "tools_hash": "the reviewed source TOOLS.md SHA-256",
  "manifest_files": 6,
  "loop_detection": true,
  "workshop_reset": true,
  "main_session_touched": false,
  "rollback_ready": true
}
```

Read back only hash, permission, manifest-key, listener, OpenClaw status, GPU identity, and absence-of-context/history checks. Confirm ports `18788`, `18789`, and `8765` remain healthy and the instance remains running. Do not inspect the user's main transcript.

- [ ] **Step 7: Run three independent four-prompt QA sessions**

For each session:

1. Generate a fresh UUID locally and use it only for this QA run.
2. Invoke the committed `scripts/run_acs_openclaw_live_qa.py` with that UUID and the accepted page. It must submit the four exact prompt blocks in order, beginning at Prompt 1.
3. Do not retry one prompt, resume a partial session, or continue after failure. On exit `75`, mark that UUID not accepted, reset only the approved workshop state, generate a different fresh UUID, and restart the complete four-prompt sequence from Prompt 1 once. A second timeout blocks that QA run. Any non-timeout failure blocks publication and enters diagnosis.
4. Export only the exact regular trajectory whose basename equals that QA UUID plus `.trajectory.jsonl` through the version-gated exact-session interface or fixed directory from Task 8 Step 1. Export the final `outputs/workshop/results.zip` to the same task-owned mode-`0700` host directory, then copy those two mode-`0600` files to the local task directory. Reject recursive search, multiple matches, a different UUID, or any path outside the fixed roots. Never enumerate or export the main session, another session, or `sessions.json`.
5. Run the committed verifier:

For run indices `1`, `2`, and `3`, set `acs_qa_index` to that integer and run:

```bash
acs_qa_index=1
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/verify_acs_openclaw_trajectory.py \
  --trajectory "/private/tmp/acs-prompt-reliability-20260821/qa-${acs_qa_index}.trajectory.jsonl" \
  --results-zip "/private/tmp/acs-prompt-reliability-20260821/qa-${acs_qa_index}-results.zip"
```

6. Require exit `0` and one closed pass receipt. Independently run `unzip -t` and a small standard-library PNG signature/dimension check for the four chat PNG members.
7. Reset only `outputs/workshop`, `context.json`, and `history.json` between sessions using the already-reviewed reset function; preserve the patched runner, `TOOLS.md`, manifest, services, loop setting, and all chat transcripts.

All three sessions must pass. One failed non-timeout session blocks publication and triggers diagnosis; it is not replaced by a cached success.

- [ ] **Step 8: Record the browser boundary honestly**

Report these gates separately:

```text
Runner and artifacts: PASS only if all three verifier receipts pass
Assistant answers: PASS only if all three exact-answer checks pass
Authenticated browser auto-login: PASS only when a fresh authenticated navigation to https://open-chemistry-agent-8id74izoa.brevlab.com/chat?session=agent%3Amain%3Amain reaches chat without a manual Gateway token or password prompt
Native chat image rendering: PASS only when all four expected MEDIA outputs render as native images in one fresh four-prompt browser session
Download Results click: PASS only when the user-visible Download Results link is clicked in that authenticated browser, downloads workshop/results.zip, and the downloaded SHA-256 equals the accepted verifier receipt
```

Mark each browser gate `NOT RUN` when its exact action was not performed, and `FAIL` when it was performed but did not meet the criterion. Do not infer a browser PASS from terminal, trajectory, HTTP, or ZIP evidence.

### Task 10: Publish accepted commits and synchronize future deployments

**Files:**
- Push: source repository branch `acs-fall-2026-launchable`
- Modify and push: `/Users/ktretina/Desktop/Codex Working Folder/digital-biology-examples/acsfall26/README.md` on `gh-pages`
- Deliver: exact generated `launchable/acs_console_bootstrap.sh` for the saved Brev Launchable definition

- [ ] **Step 1: Run final verification before publication**

Use the verification-before-completion skill. Re-run Task 8 Steps 5, 6, 7, and 8 against the exact accepted bootstrap `HEAD`, including the complete repository suite. Require the three live verifier receipts, direct ZIP/PNG checks, independent review closure, and clean Git status. Confirm the bootstrap pins the source implementation commit, not the bootstrap commit.

- [ ] **Step 2: Push source and bootstrap commits in order**

List local commits from `origin/acs-fall-2026-launchable..HEAD` and identify the final source implementation SHA and later bootstrap SHA. Push the branch once only after proving their order:

```bash
git fetch origin acs-fall-2026-launchable
git merge-base --is-ancestor origin/acs-fall-2026-launchable HEAD
acs_bootstrap_commit="$(git rev-parse HEAD)"
acs_source_commit="$(
  sed -n 's/^readonly repo_commit="\([0-9a-f]\{40\}\)"$/\1/p' \
    launchable/acs_console_bootstrap.sh
)"
test "${#acs_source_commit}" -eq 40
test "$(git rev-parse "${acs_bootstrap_commit}^")" = "${acs_source_commit}"
pinned_from_commit="$(
  git show "${acs_bootstrap_commit}:launchable/acs_console_bootstrap.sh" |
    sed -n 's/^readonly repo_commit="\([0-9a-f]\{40\}\)"$/\1/p'
)"
test "${pinned_from_commit}" = "${acs_source_commit}"
git cat-file -e "${acs_source_commit}^{commit}"
git cat-file -e "${acs_bootstrap_commit}^{commit}"
git push origin acs-fall-2026-launchable
remote_tip="$(git ls-remote origin refs/heads/acs-fall-2026-launchable | cut -f1)"
test "${remote_tip}" = "${acs_bootstrap_commit}"
```

Require the source commit to be the direct parent of the bootstrap commit, the bootstrap bytes to contain that exact source pin, and the remote branch tip to equal the local bootstrap commit. Do not force-push.

- [ ] **Step 3: Synchronize the public NVIDIA workshop page byte-for-byte**

In `/Users/ktretina/Desktop/Codex Working Folder/digital-biology-examples`, run `git fetch origin gh-pages`, verify branch `gh-pages`, require local `HEAD` to equal `origin/gh-pages`, and require clean status. Use apply-patch to replace only the four marked prompt regions in `acsfall26/README.md` with the accepted regions from the local canonical page. Do not alter unrelated public-page content.

Verify exact prompt-region identity with a small Python script that extracts all four marker regions from both files and compares their bytes. Also run:

```bash
cmp -s \
  /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/acs-fall-2026-launchable/docs/acs-fall-2026-workshop.md \
  /Users/ktretina/Desktop/Codex\ Working\ Folder/digital-biology-examples/acsfall26/README.md
shasum -a 256 \
  /Users/ktretina/Desktop/Codex\ Working\ Folder/nvmolkit-brev-notebook/.worktrees/acs-fall-2026-launchable/docs/acs-fall-2026-workshop.md \
  /Users/ktretina/Desktop/Codex\ Working\ Folder/digital-biology-examples/acsfall26/README.md
git diff --check
gitleaks git --staged --no-banner --redact .
```

Require `cmp` exit `0` and equal SHA-256 values. Because the two files were byte-identical before this change and only the four prompt regions changed locally, any other difference is a release blocker.

Commit and push:

```bash
git add acsfall26/README.md
git commit -m "docs: improve ACS chemistry agent prompts"
git push origin gh-pages
git ls-remote origin refs/heads/gh-pages
```

Require the remote tip to equal the local commit. Do not force-push.

- [ ] **Step 4: Hand off the saved Launchable update**

Measure and verify the exact generated bootstrap again:

```bash
LC_ALL=C wc -c < launchable/acs_console_bootstrap.sh
bash -n launchable/acs_console_bootstrap.sh
```

Provide the file to paste into Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`. State explicitly that the source push does not edit the saved Console definition. The user must replace only the setup-script field and preserve:

- `NVIDIA_INFERENCE_API_KEY` as the required private field;
- one NVIDIA L4 / `g6.xlarge`, 16 GiB RAM, and 128 GiB disk;
- Secure Links `Open Chemistry Agent:18788` and `Download Results:8765`;
- no public link to raw port `18789`; and
- current organization and access settings.

- [ ] **Step 5: Final evidence-backed handoff**

Report:

- exact source, bootstrap, and public-page commit SHAs;
- local test counts and static/secret gate results;
- three live session IDs hashed in the report, not printed raw;
- three safe verifier receipts and direct ZIP/PNG results;
- final instance name/ID, running state, preserved main session, ports, GPU, and remaining cost exposure;
- each browser gate as `PASS`, `FAIL`, or `NOT RUN` under its exact criterion, never inferred; and
- the saved Launchable Console bootstrap action still required from the user.

Do not claim full attendee/browser qualification until authenticated auto-login, all four native image renders, and the clicked-download SHA check all pass.

### Task 11: Execute the approved post-rollback stabilization

This task is authoritative over conflicting live-QA counts and Console-handoff
steps above.

- [ ] **Step 1: Add RED tests for the two stabilization controls**

Add focused live-operation tests that prove a hard kill during the pre-backup
`prepared` phase releases the full-process lock and permits one safe next apply,
while a concurrent same-state or different-state apply fails before backup.
Add controller tests that prove a repeated invocation ID calls the patch at
most once, stores the exact inner exit outcome, returns outer exit `0` after the
claim, and rejects malformed input before patch execution.

- [ ] **Step 2: Implement the smallest GREEN changes**

Hold one fixed non-blocking host lock for the full patch process. Reconcile only
an unambiguously pre-mutation stale `prepared` journal while holding that lock.
Add the no-replay controller with atomic claim and terminal receipt files. Do
not add general transaction, archive, metadata, or verifier features.

- [ ] **Step 3: Obtain independent spec and quality reviews**

Require no open Critical finding. Record any new non-Critical finding as a
residual risk instead of expanding scope.

- [ ] **Step 4: Run the two required final suites once**

Run, one heavy command at a time:

```bash
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_verify_acs_openclaw_trajectory.py

env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/acs-workshop-mpl \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q \
  tests/test_acs_live_instance_patch.py
```

Record the exact counts and duration. Do not rerun either complete suite during
this release decision unless the code changes again; if code changes, the old
result no longer describes final `HEAD` and the release remains blocked.

- [ ] **Step 5: Make one exact live canary decision**

Re-list organization `agents-in-ls`; verify exact instance name, ID
`8id74izoa`, state, type, GPU, shared ownership, and task namespaces. Stage the
reviewed package. Use a new private state directory and no-replay invocation ID.
Apply once, retrieve the terminal receipt read-only, and verify the installed
state. Run exactly one fresh four-prompt QA trajectory and the committed
verifier, plus direct ZIP and PNG checks. Keep the update only after every gate
passes. Otherwise roll back once through a new no-replay invocation and verify
the trusted restoration. Preserve both trusted backups.

- [ ] **Step 6: Publish only after the live pass**

Repin the generated bootstrap to the final source commit, verify its size and
syntax, and create the later bootstrap commit. Push the source branch without
force. Synchronize the accepted workshop page to `NVIDIA/digital-biology-examples`
on `gh-pages` byte-for-byte, commit, and push without force.

- [ ] **Step 7: Update and qualify one future deployment**

Use only a supported authenticated Launchable authoring surface to update exact
Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`. Preserve its secret field,
hardware, ports, organization, and access settings. If no supported surface is
callable, stop and report the exact manual field update.

After read-only confirmation of exact deployment type and price, prevent a
duplicate create and make at most one fresh deployment under the user's
approved scope. Verify fresh browser auto-login, all four native images in one
four-prompt session, and a clicked `Download Results` ZIP whose hash equals the
accepted artifact. Report each gate separately and do not infer browser results
from terminal evidence.
