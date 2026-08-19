import csv
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
MODULE2_PATH = NOTEBOOK_DIR / "02_agent_assisted_reframe_neighborhoods.ipynb"
MODULE3_PATH = NOTEBOOK_DIR / "03_full_agent_reframe_panel_design.ipynb"
WORKSHOP_AGENT_PATH = NOTEBOOK_DIR / "workshop_llm_agent.py"


def test_workshop_agent_version_matches_both_notebook_locks():
    expected_version = "2026.08.19.1"
    module2_source = "\n".join(
        cell.source for cell in nbformat.read(MODULE2_PATH, as_version=4).cells
    )
    module3_source = "\n".join(
        cell.source for cell in nbformat.read(MODULE3_PATH, as_version=4).cells
    )
    agent_source = WORKSHOP_AGENT_PATH.read_text(encoding="utf-8")

    assert f'WORKSHOP_AGENT_VERSION = "{expected_version}"' in agent_source
    assert f'EXPECTED_WORKSHOP_AGENT_VERSION = "{expected_version}"' in module2_source
    assert f'EXPECTED_WORKSHOP_AGENT_VERSION = "{expected_version}"' in module3_source


def test_module2_reference_executes_cleanly_without_key_or_hosted_client(
    monkeypatch, tmp_path
):
    notebook = nbformat.read(MODULE2_PATH, as_version=4)
    first_code = next(
        index for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
    )
    notebook.cells.insert(
        first_code,
        nbformat.v4.new_code_cell(
            f"import sys\nsys.path.insert(0, {str(NOTEBOOK_DIR)!r})\n",
            id="test-module2-import-path",
        ),
    )
    setup_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "import workshop_llm_agent" in cell.source
    )
    notebook.cells.insert(
        setup_index + 1,
        nbformat.v4.new_code_cell(
            "_blocked_client_calls = []\n"
            "def _forbidden_client(*args, **kwargs):\n"
            "    _blocked_client_calls.append((args, kwargs))\n"
            "    raise AssertionError('reference mode created a hosted client')\n"
            "_workshop_llm_agent._client = _forbidden_client\n"
            "_workshop_llm_agent.get_workshop_api_key = _forbidden_client\n",
            id="test-module2-client-blocker",
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert _blocked_client_calls == []\n",
            id="test-module2-client-assertion",
        )
    )

    matplotlib_dir = tmp_path / "matplotlib"
    ipython_dir = tmp_path / "ipython"
    matplotlib_dir.mkdir()
    ipython_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_dir))
    monkeypatch.setenv("IPYTHONDIR", str(ipython_dir))
    monkeypatch.setenv("NVMOLKIT_WORKSHOP_MODE", "reference")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    executor = ExecutePreprocessor(timeout=300, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(tmp_path)}})

    check_cell = next(cell for cell in notebook.cells if cell.id == "cell-b766b437bdb5")
    html_output = "\n".join(
        output.data.get("text/html", "")
        for output in check_cell.outputs
        if output.output_type in {"display_data", "execute_result"}
    )
    table_head = re.search(r"<thead>(.*?)</thead>", html_output, flags=re.DOTALL)
    assert table_head is not None
    table_headers = [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", header)).strip()
        for header in re.findall(
            r"<th[^>]*>(.*?)</th>", table_head.group(1), flags=re.DOTALL
        )
    ]

    output_path = tmp_path / "module2-reference-executed.ipynb"
    nbformat.write(notebook, output_path)
    assert output_path.is_file()

    stream_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert table_headers == ["", "radius", "query", "rank", "neighbor", "tanimoto"]
    assert "Normal-path invariant checks passed" in stream_text
    assert "Selected failure branches were not triggered" in stream_text
    assert "NVMOLKIT_WORKSHOP_MODE=reference" in stream_text
    assert "Implementation label: reference" in stream_text
    reference_text = stream_text.lower()
    assert "nemotron chose" not in reference_text
    assert "nemotron choose" not in reference_text
    assert "fixed local reference policy values were used" in reference_text
    assert "no hosted selection occurred" in reference_text


def _module3_notebook():
    return nbformat.read(MODULE3_PATH, as_version=4)


def _module3_code_source():
    return "\n\n".join(
        cell.source for cell in _module3_notebook().cells if cell.cell_type == "code"
    )


def _module3_cell_source(cell_id):
    return next(cell.source for cell in _module3_notebook().cells if cell.id == cell_id)


def _execute_module3_cell(cell_id, namespace):
    exec(
        compile(
            _module3_cell_source(cell_id),
            f"<03_full_agent_reframe_panel_design.ipynb:{cell_id}>",
            "exec",
        ),
        namespace,
    )


@pytest.fixture(scope="module")
def module3_reference_run(tmp_path_factory):
    run_root = tmp_path_factory.mktemp("module3-receipt-reference")
    previous_cwd = Path.cwd()
    previous_mode = os.environ.get("NVMOLKIT_WORKSHOP_MODE")
    sys.path.insert(0, str(NOTEBOOK_DIR))
    os.environ["NVMOLKIT_WORKSHOP_MODE"] = "reference"
    try:
        os.chdir(run_root)
        namespace = {}
        for cell_id in ("m3-setup", "m3-input", "m3-workspace", "m3-render"):
            _execute_module3_cell(cell_id, namespace)

        panel_agent = namespace["PanelDesignAgent"](
            workdir=namespace["AGENT_WORKDIR"],
            mission=namespace["MISSION"],
            mode="reference",
        )
        counts = {"plan": 0, "run": 0, "audit": 0}
        original_request_plan = panel_agent.request_plan
        original_run = panel_agent.run
        original_request_audit = panel_agent._request_audit

        def counted_request_plan():
            counts["plan"] += 1
            return original_request_plan()

        def counted_request_audit(*args, **kwargs):
            counts["audit"] += 1
            return original_request_audit(*args, **kwargs)

        def counted_run(**kwargs):
            counts["run"] += 1
            return original_run(**kwargs)

        panel_agent.request_plan = counted_request_plan
        panel_agent._request_audit = counted_request_audit
        panel_agent.run = counted_run
        reference_plan = panel_agent.request_plan()
        run = panel_agent.run(
            approved_strategy=reference_plan.recommended_strategy,
            expected_panel_size=namespace["PANEL_SIZE"],
            max_revisions=0,
            timeout_seconds=180,
        )
        loaded_panel, _, _, _ = namespace["load_validated_panel_artifacts"](run)
    finally:
        os.chdir(previous_cwd)
        if previous_mode is None:
            os.environ.pop("NVMOLKIT_WORKSHOP_MODE", None)
        else:
            os.environ["NVMOLKIT_WORKSHOP_MODE"] = previous_mode
        sys.path.remove(str(NOTEBOOK_DIR))

    return {
        "namespace": namespace,
        "panel_agent": panel_agent,
        "plan": reference_plan,
        "run": run,
        "counts": counts,
        "loaded_panel": loaded_panel,
    }


def _module3_receipt_namespace(reference_run, *, workdir=None):
    namespace = dict(reference_run["namespace"])
    if workdir is not None:
        namespace["AGENT_WORKDIR"] = workdir
    _execute_module3_cell("m3-render", namespace)
    return namespace


def _clone_module3_run(reference_run, destination):
    destination.mkdir()
    source_workdir = reference_run["run"].analysis_path.parent
    for name in (
        "reframe_candidates.csv",
        "analysis.py",
        "panel.csv",
        "report.json",
        "agent_trace.json",
    ):
        shutil.copy2(source_workdir / name, destination / name)
    return replace(
        reference_run["run"],
        analysis_path=destination / "analysis.py",
        panel_path=destination / "panel.csv",
        report_path=destination / "report.json",
        trace_path=destination / "agent_trace.json",
    )


def test_module3_notebook_uses_the_fixed_snapshot_and_panel_contract():
    source = _module3_code_source()

    assert "WORKSHOP_MODE = workshop_mode()" in source
    assert 'load_reframe(96, source="snapshot")' in source
    assert "PANEL_SIZE = 24" in source
    assert "len(candidate_pool) == 96" in source
    assert 'candidate_pool["canonical_ikey"].nunique() == 96' in source
    assert "REQUESTED_POOL_SIZE" not in source
    assert "effective_pool_size" not in source
    assert "/nvmolkit-brev-notebook" not in source
    assert "/.venv" not in source


def test_module3_defines_one_validated_renderer_before_either_launch_path():
    source = _module3_code_source()
    loader = source.index("def load_validated_panel_artifacts(")
    renderer = source.index("def render_validated_panel_run(")
    branch = source.index('if WORKSHOP_MODE == "reference":')

    assert loader < renderer < branch
    assert "on_complete=render_validated_panel_run" in source
    reference_branch = (
        source[branch : source.index("##", branch)]
        if "##" in source[branch:]
        else source[branch:]
    )
    assert "render_validated_panel_run(agent_run)" in reference_branch
    assert "module3_workflow.agent_run is not run" in source
    assert "Complete the interactive workflow above" not in source
    assert "Waiting for sponsor approval" in source


def test_module3_notebook_has_no_duplicate_reference_algorithm_or_stale_fallback():
    source = _module3_code_source()

    assert "def reference_panel(" not in source
    assert "USE_AGENT_OUTPUT" not in source
    assert "used_agent_output" not in source
    assert "loaded the tagged reference baseline" not in source
    assert "first 24" in source
    assert "minimum_distance" in source
    assert "descriptor_coverage" in source


def test_module3_notebook_explains_replay_uses_guardrail_language_and_three_columns():
    notebook_text = MODULE3_PATH.read_text(encoding="utf-8").lower()
    source = _module3_code_source()

    assert "approve plan & run agent" in notebook_text
    assert "rerun steps 5 and 6" in notebook_text
    assert "authoritative receipt" in notebook_text
    assert "current kernel" in notebook_text
    assert "deterministic reference audit" in notebook_text
    assert "not a hosted model call" in notebook_text
    assert "descriptor-range coverage is a guardrail" in notebook_text
    assert "minimum tanimoto distance is the strategy-sensitive" in notebook_text
    assert "molsPerRow=3" in source


def test_module3_receipt_replays_canonical_evidence_without_new_calls(
    module3_reference_run,
):
    namespace = _module3_receipt_namespace(module3_reference_run)
    replay_calls = {"plan": 0, "run": 0, "audit": 0}

    def forbidden_replay(call_name):
        def forbidden(*args, **kwargs):
            replay_calls[call_name] += 1
            raise AssertionError(f"receipt replay called {call_name}")

        return forbidden

    namespace["panel_agent"] = SimpleNamespace(
        request_plan=forbidden_replay("plan"),
        run=forbidden_replay("run"),
        _request_audit=forbidden_replay("audit"),
    )
    run = module3_reference_run["run"]
    recommended_strategy = module3_reference_run["plan"].recommended_strategy

    first = namespace["build_validated_panel_receipt"](
        run, recommended_strategy=recommended_strategy
    )
    second = namespace["build_validated_panel_receipt"](
        run, recommended_strategy=recommended_strategy
    )
    namespace["_require_canonical_run"](run, recommended_strategy)
    _, loaded_report, _, _ = namespace["_require_canonical_run"](
        run, recommended_strategy
    )

    assert first == second
    assert first["mode"] == "reference"
    assert first["model"] is None
    assert first["recommended_strategy"] == recommended_strategy
    assert first["approved_strategy"] == run.approved_strategy
    assert first["analysis_status"] == "validated"
    assert first["audit_status"] == "reference audit complete"
    assert loaded_report["parameters"]["strategy"] == "descriptor_seeded_max_min"
    assert "hosted" not in first["audit_status"]
    assert replay_calls == {"plan": 0, "run": 0, "audit": 0}
    assert module3_reference_run["counts"] == {"plan": 1, "run": 1, "audit": 1}


@pytest.mark.parametrize(
    "workflow_status", ("planning", "awaiting_approval", "executing")
)
def test_module3_pending_receipt_and_gallery_cells_are_safe_without_calls(
    workflow_status, capsys
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("pending display reached validation or execution")

    namespace = {
        "agent_run": None,
        "panel": object(),
        "PANEL_AGENT_MODE": "hosted",
        "WORKSHOP_MODE": "hosted",
        "module3_workflow": SimpleNamespace(
            status=workflow_status,
            agent_run=None,
            plan=None,
        ),
        "build_validated_panel_receipt": forbidden,
        "_require_canonical_run": forbidden,
    }
    _execute_module3_cell("m3-state", namespace)
    _execute_module3_cell("m3-gallery", namespace)

    output = capsys.readouterr().out
    waiting = "Waiting for sponsor approval. No validated result is available yet."
    assert output.count(waiting) == 2
    assert calls == []


def test_module3_plan_failed_prompts_retry_without_calls(capsys):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("plan failure reached validation or execution")

    namespace = {
        "agent_run": None,
        "PANEL_AGENT_MODE": "hosted",
        "WORKSHOP_MODE": "hosted",
        "module3_workflow": SimpleNamespace(
            status="plan_failed",
            agent_run=None,
            plan=None,
        ),
        "build_validated_panel_receipt": forbidden,
        "_require_canonical_run": forbidden,
        "display": forbidden,
        "Draw": SimpleNamespace(MolsToGridImage=forbidden),
    }

    _execute_module3_cell("m3-state", namespace)
    _execute_module3_cell("m3-gallery", namespace)

    output = capsys.readouterr().out
    retry = "Plan request failed. Use Retry Plan before approval."
    assert output.count(retry) == 2
    assert "Waiting for sponsor approval" not in output
    assert calls == []


def test_module3_reference_without_run_prompts_step4_without_calls(capsys):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("missing reference run reached validation or execution")

    namespace = {
        "agent_run": None,
        "PANEL_AGENT_MODE": "reference",
        "WORKSHOP_MODE": "reference",
        "module3_workflow": None,
        "build_validated_panel_receipt": forbidden,
        "_require_canonical_run": forbidden,
        "display": forbidden,
        "Draw": SimpleNamespace(MolsToGridImage=forbidden),
    }

    _execute_module3_cell("m3-state", namespace)
    _execute_module3_cell("m3-gallery", namespace)

    output = capsys.readouterr().out
    missing = "Reference analysis has not run. Rerun Step 4."
    assert output.count(missing) == 2
    assert "Waiting for sponsor approval" not in output
    assert calls == []


def test_module3_failed_hosted_run_is_not_reported_as_pending_or_validated(capsys):
    calls = []

    def forbidden(call_name):
        def fail(*args, **kwargs):
            calls.append((call_name, args, kwargs))
            raise AssertionError(f"failed run reached {call_name}")

        return fail

    failed_run = SimpleNamespace(success=False)
    namespace = {
        "agent_run": None,
        "module3_workflow": SimpleNamespace(
            status="failed",
            agent_run=failed_run,
            plan=SimpleNamespace(recommended_strategy=2),
        ),
        "PANEL_AGENT_MODE": "hosted",
        "WORKSHOP_MODE": "hosted",
        "build_validated_panel_receipt": forbidden("receipt validation"),
        "_require_canonical_run": forbidden("canonical validation"),
        "display": forbidden("display"),
        "Draw": SimpleNamespace(MolsToGridImage=forbidden("gallery")),
        "panel_agent": SimpleNamespace(
            request_plan=forbidden("plan"),
            run=forbidden("analysis"),
            _request_audit=forbidden("audit"),
        ),
    }

    _execute_module3_cell("m3-state", namespace)
    _execute_module3_cell("m3-gallery", namespace)

    output = capsys.readouterr().out
    assert output.count("Analysis did not validate") == 2
    assert "Waiting for sponsor approval" not in output
    assert calls == []


def test_module3_failed_hosted_workflow_without_run_is_not_reported_as_pending(
    capsys,
):
    calls = []

    def forbidden(call_name):
        def fail(*args, **kwargs):
            calls.append((call_name, args, kwargs))
            raise AssertionError(f"failed workflow reached {call_name}")

        return fail

    namespace = {
        "agent_run": None,
        "module3_workflow": SimpleNamespace(
            status="failed",
            agent_run=None,
            plan=SimpleNamespace(recommended_strategy=2),
        ),
        "PANEL_AGENT_MODE": "hosted",
        "WORKSHOP_MODE": "hosted",
        "build_validated_panel_receipt": forbidden("receipt validation"),
        "_require_canonical_run": forbidden("canonical validation"),
        "display": forbidden("display"),
        "Draw": SimpleNamespace(MolsToGridImage=forbidden("gallery")),
        "panel_agent": SimpleNamespace(
            request_plan=forbidden("plan"),
            run=forbidden("analysis"),
            _request_audit=forbidden("audit"),
        ),
    }

    _execute_module3_cell("m3-state", namespace)
    _execute_module3_cell("m3-gallery", namespace)

    output = capsys.readouterr().out
    assert output.count("Analysis did not validate") == 2
    assert "Waiting for sponsor approval" not in output
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "different workflow run object",
        "wrong fixed path",
        "missing trace",
        "wrong trace mode",
        "wrong trace model",
        "wrong approved strategy",
        "trace success false",
        "wrong trace recommendation",
        "run and trace audit disagreement",
        "analysis source tamper",
        "candidate input tamper",
    ),
)
def test_module3_receipt_rejects_inconsistent_retained_evidence(
    mutation, module3_reference_run, tmp_path
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    recommended_strategy = module3_reference_run["plan"].recommended_strategy
    trace_path = workdir / "agent_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    if mutation == "different workflow run object":
        namespace["PANEL_AGENT_MODE"] = "hosted"
        namespace["WORKSHOP_MODE"] = "hosted"
        namespace["module3_workflow"] = SimpleNamespace(
            status="completed",
            agent_run=object(),
            plan=SimpleNamespace(recommended_strategy=recommended_strategy),
        )
    elif mutation == "wrong fixed path":
        run = replace(run, analysis_path=workdir / "unexpected-analysis.py")
    elif mutation == "missing trace":
        trace_path.unlink()
    elif mutation == "wrong trace mode":
        trace["mode"] = "hosted"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "wrong trace model":
        trace["model"] = "nvidia/unexpected-model"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "wrong approved strategy":
        trace["approved_strategy"] = 1 if run.approved_strategy == 2 else 2
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "trace success false":
        trace["success"] = False
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "wrong trace recommendation":
        trace["plan"]["recommended_strategy"] = 1 if recommended_strategy == 2 else 2
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "run and trace audit disagreement":
        trace["audit"] = None
        trace["audit_error"] = ""
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
    elif mutation == "analysis source tamper":
        (workdir / "analysis.py").write_text(
            "print('tampered source')\n", encoding="utf-8"
        )
    elif mutation == "candidate input tamper":
        candidate_path = workdir / "reframe_candidates.csv"
        panel_path = workdir / "panel.csv"
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            candidate_rows = list(csv.DictReader(handle))
        with panel_path.open(newline="", encoding="utf-8") as handle:
            panel_rows = list(csv.DictReader(handle))
        selected_key = panel_rows[0]["canonical_ikey"]
        tampered_name = "Tampered selected compound"
        next(row for row in candidate_rows if row["canonical_ikey"] == selected_key)[
            "name"
        ] = tampered_name
        panel_rows[0]["name"] = tampered_name
        with candidate_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=candidate_rows[0])
            writer.writeheader()
            writer.writerows(candidate_rows)
        with panel_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=panel_rows[0])
            writer.writeheader()
            writer.writerows(panel_rows)

    visible_output = []
    namespace.update(
        agent_run=run,
        panel=module3_reference_run["loaded_panel"],
        reference_plan=module3_reference_run["plan"],
        print=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        display=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        Draw=SimpleNamespace(
            MolsToGridImage=lambda *args, **kwargs: visible_output.append(
                (args, kwargs)
            )
        ),
    )

    with pytest.raises(ValueError):
        _execute_module3_cell("m3-state", namespace)
    with pytest.raises(ValueError):
        _execute_module3_cell("m3-gallery", namespace)
    assert visible_output == []


def test_module3_receipt_uses_report_snapshot_from_same_validation(
    module3_reference_run, tmp_path, monkeypatch
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    agent_module = namespace["_workshop_llm_agent"]
    original_snapshot_validator = agent_module._validate_panel_artifacts_snapshot
    report_path = workdir / "report.json"
    expected_backend = json.loads(report_path.read_text(encoding="utf-8"))["backend"]

    def validate_then_tamper(*args, **kwargs):
        receipt, report_snapshot = original_snapshot_validator(*args, **kwargs)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["backend"] = "tampered-after-validation"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return receipt, report_snapshot

    monkeypatch.setattr(
        agent_module, "_validate_panel_artifacts_snapshot", validate_then_tamper
    )

    receipt = namespace["build_validated_panel_receipt"](
        run,
        recommended_strategy=module3_reference_run["plan"].recommended_strategy,
    )

    assert receipt["backend"] == expected_backend
    assert json.loads(report_path.read_text(encoding="utf-8"))["backend"] == (
        "tampered-after-validation"
    )


def test_module3_receipt_rejects_panel_changed_after_validation(
    module3_reference_run, tmp_path, monkeypatch
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    agent_module = namespace["_workshop_llm_agent"]
    original_snapshot_validator = agent_module._validate_panel_artifacts_snapshot
    panel_path = workdir / "panel.csv"

    def validate_then_replace_panel(*args, **kwargs):
        receipt, report_snapshot = original_snapshot_validator(*args, **kwargs)
        with panel_path.open(newline="", encoding="utf-8") as handle:
            panel_rows = list(csv.DictReader(handle))
        panel_rows[0]["name"] = "Unvalidated replacement chemistry"
        with panel_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=panel_rows[0])
            writer.writeheader()
            writer.writerows(panel_rows)
        return receipt, report_snapshot

    monkeypatch.setattr(
        agent_module,
        "_validate_panel_artifacts_snapshot",
        validate_then_replace_panel,
    )
    visible_output = []
    namespace.update(
        agent_run=run,
        reference_plan=module3_reference_run["plan"],
        print=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        display=lambda *args, **kwargs: visible_output.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="changed during validation"):
        _execute_module3_cell("m3-state", namespace)
    assert visible_output == []


def test_module3_receipt_rejects_candidate_changed_during_validation(
    module3_reference_run, tmp_path, monkeypatch
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    agent_module = namespace["_workshop_llm_agent"]
    original_snapshot_validator = agent_module._validate_panel_artifacts_snapshot
    candidate_path = workdir / "reframe_candidates.csv"

    def validate_then_replace_candidate(*args, **kwargs):
        receipt, report_snapshot = original_snapshot_validator(*args, **kwargs)
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            candidate_rows = list(csv.DictReader(handle))
        candidate_rows[0]["name"] = "Unvalidated replacement candidate"
        with candidate_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=candidate_rows[0])
            writer.writeheader()
            writer.writerows(candidate_rows)
        return receipt, report_snapshot

    monkeypatch.setattr(
        agent_module,
        "_validate_panel_artifacts_snapshot",
        validate_then_replace_candidate,
    )
    visible_output = []
    namespace.update(
        agent_run=run,
        reference_plan=module3_reference_run["plan"],
        print=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        display=lambda *args, **kwargs: visible_output.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="reframe_candidates.csv changed"):
        _execute_module3_cell("m3-state", namespace)
    assert visible_output == []


def test_module3_receipt_rejects_analysis_changed_during_validation(
    module3_reference_run, tmp_path, monkeypatch
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    agent_module = namespace["_workshop_llm_agent"]
    original_snapshot_validator = agent_module._validate_panel_artifacts_snapshot
    analysis_path = workdir / "analysis.py"

    def validate_then_replace_analysis(*args, **kwargs):
        receipt, report_snapshot = original_snapshot_validator(*args, **kwargs)
        analysis_path.write_text("print('unvalidated source')\n", encoding="utf-8")
        return receipt, report_snapshot

    monkeypatch.setattr(
        agent_module,
        "_validate_panel_artifacts_snapshot",
        validate_then_replace_analysis,
    )
    visible_output = []
    namespace.update(
        agent_run=run,
        reference_plan=module3_reference_run["plan"],
        print=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        display=lambda *args, **kwargs: visible_output.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="analysis.py changed"):
        _execute_module3_cell("m3-state", namespace)
    assert visible_output == []


def test_module3_receipt_rejects_report_strategy_that_disagrees_with_run(
    module3_reference_run, tmp_path
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    original_loader = namespace["load_validated_panel_artifacts"]

    def load_with_wrong_strategy(run_to_load):
        panel, report, trace, receipt = original_loader(run_to_load)
        changed_report = json.loads(json.dumps(report))
        changed_report["parameters"]["strategy"] = "cluster_aware_max_min"
        return panel, changed_report, trace, receipt

    namespace["load_validated_panel_artifacts"] = load_with_wrong_strategy

    with pytest.raises(ValueError, match="reported strategy"):
        namespace["_require_canonical_run"](
            run, module3_reference_run["plan"].recommended_strategy
        )


def test_module3_immediate_renderer_validates_canonical_run_once(
    module3_reference_run, monkeypatch
):
    namespace = _module3_receipt_namespace(module3_reference_run)
    original_require = namespace["_require_canonical_run"]
    validation_calls = []

    def counted_require(run, recommended_strategy):
        validation_calls.append((run, recommended_strategy))
        return original_require(run, recommended_strategy)

    namespace.update(
        _require_canonical_run=counted_require,
        reference_plan=module3_reference_run["plan"],
        display=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(namespace["plt"], "show", lambda: None)

    namespace["render_validated_panel_run"](module3_reference_run["run"])

    assert validation_calls == [
        (
            module3_reference_run["run"],
            module3_reference_run["plan"].recommended_strategy,
        )
    ]


def test_module3_receipt_revalidates_report_before_using_stale_panel(
    module3_reference_run, tmp_path
):
    workdir = tmp_path / "module3_agent_workspace"
    run = _clone_module3_run(module3_reference_run, workdir)
    report_path = workdir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["acceptance"]["selected_minimum_distance"] += 0.1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    visible_output = []
    namespace = _module3_receipt_namespace(module3_reference_run, workdir=workdir)
    namespace.update(
        agent_run=run,
        panel=module3_reference_run["loaded_panel"],
        reference_plan=module3_reference_run["plan"],
        print=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        display=lambda *args, **kwargs: visible_output.append((args, kwargs)),
        Draw=SimpleNamespace(
            MolsToGridImage=lambda *args, **kwargs: visible_output.append(
                (args, kwargs)
            )
        ),
    )
    validation_error = namespace["_workshop_llm_agent"].WorkshopAgentError

    with pytest.raises(validation_error, match="independent validation"):
        _execute_module3_cell("m3-state", namespace)
    with pytest.raises(validation_error, match="independent validation"):
        _execute_module3_cell("m3-gallery", namespace)
    assert visible_output == []


def test_module3_reference_executes_cleanly_without_key_client_or_network(
    monkeypatch, tmp_path
):
    notebook = _module3_notebook()
    state_source = next(cell.source for cell in notebook.cells if cell.id == "m3-state")
    gallery_source = next(
        cell.source for cell in notebook.cells if cell.id == "m3-gallery"
    )
    first_code = next(
        index for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
    )
    notebook.cells.insert(
        first_code,
        nbformat.v4.new_code_cell(
            f"import sys\nsys.path.insert(0, {str(NOTEBOOK_DIR)!r})\n",
            id="test-module3-import-path",
        ),
    )
    setup_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "import workshop_llm_agent" in cell.source
    )
    notebook.cells.insert(
        setup_index + 1,
        nbformat.v4.new_code_cell(
            "_blocked_client_calls = []\n"
            "_blocked_network_attempts = []\n"
            "_original_read_csv = pd.read_csv\n"
            "def _forbidden_client(*args, **kwargs):\n"
            "    _blocked_client_calls.append((args, kwargs))\n"
            "    raise AssertionError('reference mode created a hosted client')\n"
            "def _local_only_read_csv(source, *args, **kwargs):\n"
            "    if str(source).startswith(('http://', 'https://')):\n"
            "        _blocked_network_attempts.append(str(source))\n"
            "        raise AssertionError('reference mode attempted network access')\n"
            "    return _original_read_csv(source, *args, **kwargs)\n"
            "_workshop_llm_agent._client = _forbidden_client\n"
            "_workshop_llm_agent.get_workshop_api_key = _forbidden_client\n"
            "pd.read_csv = _local_only_read_csv\n"
            "_module3_plan_calls = 0\n"
            "_module3_run_calls = 0\n"
            "_module3_audit_calls = 0\n"
            "_original_panel_request_plan = PanelDesignAgent.request_plan\n"
            "_original_panel_run = PanelDesignAgent.run\n"
            "_original_panel_request_audit = PanelDesignAgent._request_audit\n"
            "def _count_panel_request_plan(self, *args, **kwargs):\n"
            "    global _module3_plan_calls\n"
            "    _module3_plan_calls += 1\n"
            "    return _original_panel_request_plan(self, *args, **kwargs)\n"
            "def _count_panel_run(self, *args, **kwargs):\n"
            "    global _module3_run_calls\n"
            "    _module3_run_calls += 1\n"
            "    return _original_panel_run(self, *args, **kwargs)\n"
            "def _count_panel_request_audit(self, *args, **kwargs):\n"
            "    global _module3_audit_calls\n"
            "    _module3_audit_calls += 1\n"
            "    return _original_panel_request_audit(self, *args, **kwargs)\n"
            "PanelDesignAgent.request_plan = _count_panel_request_plan\n"
            "PanelDesignAgent.run = _count_panel_run\n"
            "PanelDesignAgent._request_audit = _count_panel_request_audit\n",
            id="test-module3-client-blocker",
        ),
    )
    notebook.cells.extend(
        [
            nbformat.v4.new_code_cell(
                state_source,
                id="test-module3-state-replay",
            ),
            nbformat.v4.new_code_cell(
                gallery_source,
                id="test-module3-gallery-replay",
            ),
        ]
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert _blocked_client_calls == []\n"
            "assert _blocked_network_attempts == []\n"
            "assert _module3_plan_calls == 1\n"
            "assert _module3_run_calls == 1\n"
            "assert _module3_audit_calls == 1\n",
            id="test-module3-client-assertion",
        )
    )

    matplotlib_dir = tmp_path / "matplotlib"
    ipython_dir = tmp_path / "ipython"
    matplotlib_dir.mkdir()
    ipython_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_dir))
    monkeypatch.setenv("IPYTHONDIR", str(ipython_dir))
    monkeypatch.setenv("NVMOLKIT_WORKSHOP_MODE", "reference")
    monkeypatch.setenv(
        "REFRAME_CSV", "https://hostile.invalid/reframe.csv?token=do-not-disclose"
    )
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    executor = ExecutePreprocessor(timeout=300, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(tmp_path)}})

    output_path = tmp_path / "module3-reference-executed.ipynb"
    nbformat.write(notebook, output_path)
    assert output_path.is_file()

    workspace = tmp_path / "module3_agent_workspace"
    for name in ("analysis.py", "panel.csv", "report.json", "agent_trace.json"):
        assert (workspace / name).is_file()

    stream_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    match = re.search(r"^MODULE3_REPORT_JSON=(\{.*\})$", stream_text, re.MULTILINE)
    assert match is not None
    report = json.loads(match.group(1))
    assert report["mode"] == "reference"
    assert report["model"] is None
    assert report["recommended_strategy"] == 2
    assert report["approved_strategy"] == 2
    assert report["backend"] in {
        "nvmolkit-gpu",
        "rdkit-cpu-reference (not GPU evidence)",
    }
    assert report["analysis_status"] == "validated"
    assert report["audit_status"] == "reference audit complete"
    assert report["acceptance_passed"] is True

    rich_outputs = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type in {"display_data", "execute_result"}
    ]
    assert any(
        "image/png" in output.get("data", {})
        or "image/svg+xml" in output.get("data", {})
        for output in rich_outputs
    )
