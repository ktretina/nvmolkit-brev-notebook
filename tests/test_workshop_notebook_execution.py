import json
import re
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
MODULE2_PATH = NOTEBOOK_DIR / "02_agent_assisted_reframe_neighborhoods.ipynb"
MODULE3_PATH = NOTEBOOK_DIR / "03_full_agent_reframe_panel_design.ipynb"


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
    assert "module3_workflow.agent_run" not in source
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


def test_module3_reference_executes_cleanly_without_key_client_or_network(
    monkeypatch, tmp_path
):
    notebook = _module3_notebook()
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
            "pd.read_csv = _local_only_read_csv\n",
            id="test-module3-client-blocker",
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert _blocked_client_calls == []\n"
            "assert _blocked_network_attempts == []\n",
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
    assert report["candidate_count"] == 96
    assert report["panel_count"] == 24
    assert report["strict_subset"] is True
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
