import ast
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nvmolkit_nemotron_demo.ipynb"
)
REPO_ROOT = NOTEBOOK_PATH.parents[1]

STORY_HEADINGS = [
    "# nvMolKit + Nemotron",
    "## 1. Preflight",
    "## 2. Molecular sample",
    "## 3. Nemotron plan",
    "## 4. Fingerprints, similarity, and clusters",
    "## 5. Conformers and MMFF94",
    "## 6. What the results mean",
]

REQUIRED_CALLS = {
    "MorganFingerprintGenerator",
    "crossTanimotoSimilarity",
    "fused_butina",
    "EmbedMolecules",
    "MMFFOptimizeMoleculesConfs",
    "Point3D",
    "torch.cuda.synchronize",
    "Draw.MolsToGridImage",
    "sns.heatmap",
    "py3Dmol.view",
}


def read_notebook():
    stored = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert stored["nbformat"] == 4

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=nbformat.NO_CONVERT)
    nbformat.validate(notebook)
    return notebook


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_notebook_has_exact_v4_story_structure_and_is_only_source_notebook():
    notebook = read_notebook()
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]
    tracked_notebooks = subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_notebooks = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts and "executed" not in path.parts
    )
    intro = notebook.cells[0].source
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert headings == STORY_HEADINGS
    assert tracked_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    assert source_notebooks == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    assert all(
        phrase in intro
        for phrase in ("API entry-point map", "runtime requirements", "recipes", "boundaries")
    )
    assert all(
        requirement in setup
        for requirement in (
            "sys.implementation.name",
            "(3, 12)",
            '"Linux"',
            '"x86_64"',
        )
    )
    assert "Runtime: Linux x86-64 with CPython 3.12" in fields
    assert "Linux x86-64 with CPython 3.12" in readme
    assert "qualified for a fresh launch" not in fields.lower()
    assert "qualified for a fresh launch" not in readme.lower()
    assert "not yet live-qualified" in fields
    assert "not yet live-qualified" in readme
    assert all(
        acceptance in fields.lower() and acceptance in readme.lower()
        for acceptance in ("gpu", "hosted inference", "rendered visuals", "secure link")
    )
    assert ".jupyter.pid" in gitignore


def test_notebook_code_calls_required_workflow_and_visuals():
    notebook = read_notebook()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    tree = ast.parse(code)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {dotted_name(node.func) for node in calls}
    fused_butina_call = next(
        call for call in calls if dotted_name(call.func) == "fused_butina"
    )
    mmff_call = next(
        call for call in calls if dotted_name(call.func) == "MMFFOptimizeMoleculesConfs"
    )
    mmff_keywords = {keyword.arg: keyword.value for keyword in mmff_call.keywords}
    referenced_names = {
        dotted_name(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert REQUIRED_CALLS <= called_names
    assert len(fused_butina_call.args) == 1
    assert isinstance(fused_butina_call.args[0], ast.Call)
    assert dotted_name(fused_butina_call.args[0].func) == "fingerprints.torch"
    assert dotted_name(mmff_keywords["output"]) == "CoordinateOutput.DEVICE"
    assert {
        "optimization_result.energies.numpy",
        "optimization_result.converged.numpy",
        "optimization_result.mol_indices.numpy",
        "optimization_result.conf_indices.numpy",
        "optimization_result.per_molecule",
    } <= called_names
    assert {
        "optimization_result.energies",
        "optimization_result.converged",
        "optimization_result.mol_indices",
        "optimization_result.conf_indices",
        "optimization_result.per_molecule",
    } <= referenced_names
    assert "converged_conformers" in string_literals
    assert {
        "requested_conformers",
        "generated_conformers",
        "mmff_attempted_conformers",
    } <= string_literals
    assert "optimized_conformers" not in string_literals

    requested_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "requested_conformers"
            for target in node.targets
        )
    )
    embed_call = next(call for call in calls if dotted_name(call.func) == "EmbedMolecules")
    assert requested_assignment.lineno < embed_call.lineno

    explanation_call = next(
        call for call in calls if dotted_name(call.func) == "request_explanation"
    )
    explanation_line = explanation_call.lineno
    lines = code.splitlines()
    before_explanation = "\n".join(lines[: explanation_line - 1])
    after_explanation = "\n".join(lines[explanation_line:])
    boundary_terms = (
        "binding",
        "activity",
        "ADMET",
        "efficacy",
        "safety",
        "synthesizability",
        "clinical relevance",
        "experimentally validated conformations",
    )
    assert all(term in before_explanation for term in boundary_terms)
    assert "Agent-generated interpretation; verify independently" in before_explanation
    assert all(term in after_explanation for term in boundary_terms)


def test_notebook_contains_no_api_key_and_no_saved_execution_state():
    notebook = read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "nvapi-" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None


def test_setup_uses_brev_managed_python_and_leaves_jupyter_to_brev():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert setup.splitlines()[0] == "#!/bin/bash"
    assert '${HOME}/.venv/bin/python3' in setup
    assert "command -v python3.12" not in setup
    assert '"${PYTHON}" -m pip --version' in setup
    assert '"${PYTHON}" -m ensurepip --upgrade' in setup
    assert '"${PYTHON}" -m pip install --upgrade pip' in setup
    assert '"${PYTHON}" -m pip install -r requirements.txt' in setup
    assert '"${PYTHON}" - <<\'PY\'' in setup
    assert 'url = "http://127.0.0.1:8888/api"' in setup
    assert "/api/status" not in setup
    assert "while time.monotonic() < deadline" in setup
    assert "urllib.request.urlopen" in setup
    assert "response.geturl()" in setup
    assert "json.load(response)" in setup
    assert all(
        forbidden not in setup
        for forbidden in ("jupyter lab", "nohup", "PID_FILE", "kill ", "-m venv")
    )
    assert "jupyterlab==" not in requirements


def test_setup_runs_only_managed_runtime_and_probes_existing_jupyter(tmp_path):
    fake_home = tmp_path / "home"
    managed_bin = fake_home / ".venv" / "bin"
    fake_bin = tmp_path / "bin"
    managed_bin.mkdir(parents=True)
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations.log"

    managed_python = managed_bin / "python3"
    managed_python.write_text(
        """#!/bin/bash
set -u
case "${1:-}" in
  -c)
    printf 'VERSION_CHECK %s\n' "${2:-}" >>"${INVOCATION_LOG}"
    ;;
  --version)
    printf 'Python 3.12.13\n'
    ;;
  -m)
    printf 'MODULE %s %s %s\n' "${2:-}" "${3:-}" "${4:-}" >>"${INVOCATION_LOG}"
    if [[ "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
      exit 1
    fi
    ;;
  -)
    payload="$(</dev/stdin)"
    if [[ "${payload}" == *"torch.cuda.is_available"* ]]; then
      printf 'SMOKE\n' >>"${INVOCATION_LOG}"
    elif [[ "${payload}" == *"urllib.request.urlopen"* ]]; then
      printf 'HEALTH\n' >>"${INVOCATION_LOG}"
    else
      printf 'UNEXPECTED_STDIN\n' >>"${INVOCATION_LOG}"
      exit 91
    fi
    ;;
  *)
    printf 'UNEXPECTED %s\n' "$*" >>"${INVOCATION_LOG}"
    exit 92
    ;;
esac
""",
        encoding="utf-8",
    )
    managed_python.chmod(0o755)

    (fake_bin / "python3.12").write_text(
        """#!/bin/bash
printf 'FALLBACK %s\n' "$*" >>"${INVOCATION_LOG}"
exit 99
""",
        encoding="utf-8",
    )
    (fake_bin / "python3.12").chmod(0o755)
    (fake_bin / "uname").write_text(
        """#!/bin/bash
case "${1:-}" in
  -s) printf 'Linux\n' ;;
  -m) printf 'x86_64\n' ;;
  *) exit 93 ;;
esac
""",
        encoding="utf-8",
    )
    (fake_bin / "uname").chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "INVOCATION_LOG": str(invocation_log),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "launchable" / "setup.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert any("sys.implementation.name" in line for line in invocations)
    assert "MODULE pip --version " in invocations
    ensurepip_index = invocations.index("MODULE ensurepip --upgrade ")
    pip_upgrade_index = invocations.index("MODULE pip install --upgrade")
    requirements_index = invocations.index("MODULE pip install -r")
    smoke_index = invocations.index("SMOKE")
    assert "HEALTH" in invocations
    health_index = invocations.index("HEALTH")
    assert ensurepip_index < pip_upgrade_index < requirements_index < smoke_index < health_index
    assert all(not line.startswith("FALLBACK") for line in invocations)


def health_probe_source():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    start = setup.index("import json\nimport time\nimport urllib.error")
    return setup[start : setup.index("\nPY", start)]


def run_health_probe(server_mode):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if server_mode == "redirect":
                if self.path == "/api":
                    self.send_response(302)
                    self.send_header("Location", "/login")
                    self.end_headers()
                else:
                    body = b"<html>login</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return

            if self.path != "/api":
                self.send_error(404)
                return
            body = b'{"version": "4.4.5"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe = re.sub(
            r'url = "http://127\.0\.0\.1:8888/[^"]+"',
            f'url = "http://127.0.0.1:{server.server_port}/api"',
            health_probe_source(),
        )
        probe = probe.replace(
            "deadline = time.monotonic() + 60",
            "deadline = time.monotonic() + 0.2",
        ).replace("time.sleep(1)", "time.sleep(0.01)")
        return subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_health_probe_rejects_redirect_to_login_html():
    result = run_health_probe("redirect")

    assert result.returncode != 0
    assert "did not become healthy" in result.stderr


def test_health_probe_accepts_versioned_api_json():
    result = run_health_probe("valid")

    assert result.returncode == 0, result.stderr
    assert "health probe passed" in result.stdout


def test_notebook_prompts_for_missing_api_key_without_exposing_it():
    notebook = read_notebook()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    tree = ast.parse(code)
    getpass_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "getpass"
    ]
    assert len(getpass_calls) == 1
    getpass_call = getpass_calls[0]
    getpass_if = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and getpass_call in list(ast.walk(node))
    )

    assert 'os.environ.get("NVIDIA_API_KEY"' in code
    assert 'os.environ["NVIDIA_API_KEY"]' not in code
    assert "NEMOTRON_MODEL" not in code
    assert 'model = "nvidia/nemotron-3-nano-30b-a3b"' in code
    assert getpass_call.args[0].value == "NVIDIA API key (input hidden): "
    assert any(
        isinstance(node, ast.Raise) for node in ast.walk(getpass_if)
    ), "The missing-key fallback must reject empty input."
    assert "api_key" not in {
        dotted_name(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "print"
        for arg in node.args
    }


def test_launch_instructions_use_brev_managed_jupyter_and_hidden_key_prompt():
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for instructions in (fields, readme):
        assert "Brev-managed Jupyter" in instructions
        assert "Enable Jupyter" in instructions
        assert "hidden" in instructions.lower()
        assert "not rely on setup-variable persistence" in instructions
        assert "NEMOTRON_MODEL" not in instructions

    assert "hidden notebook prompt" in read_notebook().cells[1].source
