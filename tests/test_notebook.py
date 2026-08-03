import ast
import hashlib
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nvmolkit_nemotron_demo.ipynb"
HEADINGS = (
    "# Can an AI chemistry agent use nvMolKit?",
    "## Preflight",
    "## User request",
    "## Agent run",
    "## Boundary",
)
NVMOLKIT_ENTRY_POINTS = (
    "MorganFingerprintGenerator",
    "crossTanimotoSimilarity",
    "fused_butina",
    "EmbedMolecules",
    "MMFFOptimizeMoleculesConfs",
)


def read_notebook():
    stored = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert stored["nbformat"] == 4
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=nbformat.NO_CONVERT)
    nbformat.validate(notebook)
    return notebook


def notebook_code(notebook):
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_notebook_is_one_clean_eight_cell_story():
    notebook = read_notebook()
    assert len(notebook.cells) == 8
    assert [cell.cell_type for cell in notebook.cells] == [
        "markdown", "markdown", "code", "markdown", "code", "markdown", "code", "markdown"
    ]
    assert tuple(
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ) == HEADINGS
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert sum(len(cell.source.splitlines()) for cell in code_cells) <= 25
    for cell in code_cells:
        ast.parse(cell.source)
        assert cell.execution_count is None
        assert cell.outputs == []


def test_notebook_exposes_one_goal_and_one_public_agent_call():
    notebook = read_notebook()
    code = notebook_code(notebook)
    tree = ast.parse(code)
    assert sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)) == 0
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "USER_GOAL" for target in node.targets)
    ]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and dotted_name(node.func) == "run_workflow"]
    assert len(assignments) == 1
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == "run_workflow(USER_GOAL, api_key, display_events=True)"
    assert "result = run_workflow(USER_GOAL, api_key, display_events=True)" in notebook.cells[6].source
    assert all(forbidden not in code for forbidden in (
        "read_nvmolkit_skill", "prepare_molecular_sample", "compute_morgan_fingerprints",
        "compute_tanimoto_similarity", "cluster_with_fused_butina",
        "generate_and_optimize_conformers", "py3Dmol",
    ))


def test_intro_and_run_text_make_attribution_and_grounding_explicit():
    notebook = read_notebook()
    intro = notebook.cells[0].source
    run_text = notebook.cells[5].source
    combined = f"{intro}\n{run_text}"
    assert "pinned BioNeMo Agent Toolkit nvMolKit skill" in intro
    assert "initial context" in intro
    assert "one persistent Nemotron conversation" in intro
    assert "fixed dependency chain" in intro
    assert "RDKit" in combined and "input" in combined
    assert "Python" in combined and "canonical evidence" in combined
    for entry_point in NVMOLKIT_ENTRY_POINTS:
        assert entry_point in combined
    assert "six sequential validated executions" in run_text
    assert "evidence-linked, schema-checked synthesis" in run_text
    assert "qualitative interpretation is not automatically fact-verified" in run_text


def test_preflight_is_fixed_gpu_ready_and_credential_safe():
    notebook = read_notebook()
    code = notebook.cells[2].source
    tree = ast.parse(code)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imports == {"Path", "notebook_preflight", "run_workflow"}
    assert "PROJECT_ROOT / \"demo_agent.py\"" in code
    assert "api_key = notebook_preflight()" in code
    assert len(code.splitlines()) <= 10
    assert all(token not in code for token in ("torch", "getpass", "NVIDIA_API_KEY", "nvmolkit"))


def test_boundary_is_scientifically_bounded_and_claim_safe():
    boundary = read_notebook().cells[7].source.lower()
    for phrase in (
        "bounded fixed workflow", "binding", "activity", "admet", "safety",
        "within-molecule", "not global", "experimental", "no performance claims",
        "schema, evidence references, and exact rendered metrics",
        "qualitative interpretation is not automatically fact-verified",
    ):
        assert phrase in boundary


def test_fixed_artifacts_and_skill_provenance_are_intact():
    assert subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines() == ["notebooks/nvmolkit_nemotron_demo.ipynb"]
    provenance = (REPO_ROOT / "skills" / "nvmolkit" / "PROVENANCE.md").read_text(encoding="utf-8")
    skill_bytes = (REPO_ROOT / "skills" / "nvmolkit" / "SKILL.md").read_bytes()
    assert "ce151c15470991c8cb9a0efdd531a124c346ca5b" in provenance
    assert hashlib.sha256(skill_bytes).hexdigest() in provenance
    for relative_path in ("requirements.txt", "launchable/setup.sh", "launchable/fields.md", "data/sample_molecules.csv"):
        committed = subprocess.run(
            ["git", "show", f"31c8567b6ed743e56e87ee3475b4c143a7614c9b:{relative_path}"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        ).stdout
        assert (REPO_ROOT / relative_path).read_bytes() == committed


def test_readme_preserves_launch_and_separate_acceptance_gates():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    for instructions in (readme, fields):
        assert "Brev-managed Jupyter" in instructions
        assert "Only my organization" in instructions
        assert "Secure Link" in instructions
        assert "CPython 3.12" in instructions
        assert "hidden" in instructions.lower()
        assert "not rely on setup-variable persistence" in instructions
    lowered = readme.lower()
    for gate in ("local deterministic acceptance", "gpu acceptance", "hosted inference acceptance", "rendered deployment acceptance"):
        assert gate in lowered
    assert "one plan" in lowered and "six validated executions" in lowered
    assert "one evidence-linked, schema-checked synthesis" in lowered
    assert "qualitative interpretation is not automatically fact-verified" in lowered
    assert "not yet live-qualified" in lowered


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
    assert 'url = "http://127.0.0.1:8888/api"' in setup
    assert all(forbidden not in setup for forbidden in ("jupyter lab", "nohup", "PID_FILE", "kill ", "-m venv"))
    assert "jupyterlab==" not in requirements


def test_setup_runs_only_managed_runtime_and_probes_existing_jupyter(tmp_path):
    fake_home = tmp_path / "home"
    managed_bin = fake_home / ".venv" / "bin"
    fake_bin = tmp_path / "bin"
    managed_bin.mkdir(parents=True)
    fake_bin.mkdir()
    log = tmp_path / "invocations.log"
    managed_python = managed_bin / "python3"
    managed_python.write_text("""#!/bin/bash
case "${1:-}" in
  -c) printf 'VERSION_CHECK %s\n' "${2:-}" >>"${INVOCATION_LOG}" ;;
  --version) printf 'Python 3.12.13\n' ;;
  -m) printf 'MODULE %s %s %s\n' "${2:-}" "${3:-}" "${4:-}" >>"${INVOCATION_LOG}"; [[ "${2:-} ${3:-}" == "pip --version" ]] && exit 1 || true ;;
  -) payload="$(</dev/stdin)"; [[ "$payload" == *"torch.cuda.is_available"* ]] && printf 'SMOKE\n' >>"${INVOCATION_LOG}" || printf 'HEALTH\n' >>"${INVOCATION_LOG}" ;;
  *) exit 92 ;;
esac
""", encoding="utf-8")
    managed_python.chmod(0o755)
    (fake_bin / "uname").write_text("""#!/bin/bash
[[ "${1:-}" == "-s" ]] && printf 'Linux\n' || printf 'x86_64\n'
""", encoding="utf-8")
    (fake_bin / "uname").chmod(0o755)
    env = os.environ | {"HOME": str(fake_home), "INVOCATION_LOG": str(log), "PATH": f"{fake_bin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(REPO_ROOT / "launchable" / "setup.sh")], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert any("sys.implementation.name" in line for line in invocations)
    assert invocations.index("MODULE ensurepip --upgrade ") < invocations.index("MODULE pip install --upgrade")
    assert invocations.index("MODULE pip install --upgrade") < invocations.index("MODULE pip install -r")
    assert invocations.index("MODULE pip install -r") < invocations.index("SMOKE") < invocations.index("HEALTH")


def health_probe_source():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    start = setup.index("import json\nimport time\nimport urllib.error")
    return setup[start:setup.index("\nPY", start)]


def run_health_probe(mode):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if mode == "redirect":
                self.send_response(302 if self.path == "/api" else 200)
                if self.path == "/api":
                    self.send_header("Location", "/login")
                self.end_headers()
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
        probe = re.sub(r'url = "http://127\.0\.0\.1:8888/[^"]+"', f'url = "http://127.0.0.1:{server.server_port}/api"', health_probe_source())
        probe = probe.replace("deadline = time.monotonic() + 60", "deadline = time.monotonic() + 0.2").replace("time.sleep(1)", "time.sleep(0.01)")
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=2)
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
