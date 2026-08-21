from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shlex
import sys
import textwrap
import traceback
from types import ModuleType, SimpleNamespace
import ast

import ipywidgets as widgets
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "notebooks" / "workshop_llm_agent.py"
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "02_agent_assisted_reframe_neighborhoods.ipynb"
)
MODULE3_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "03_full_agent_reframe_panel_design.ipynb"
)
WORKFLOW_PATH = REPO_ROOT / "notebooks" / "module3_interactive_workflow.py"
SNAPSHOT_PATH = REPO_ROOT / "notebooks" / "data" / "reframe_teaching_snapshot.csv"
SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-18-live-browser-notebook-experience-design.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-08-18-live-browser-notebook-experience.md"
)


def _notebook_cell_source(path: Path, cell_id: str) -> str:
    stored = json.loads(path.read_text(encoding="utf-8"))
    cell = next(cell for cell in stored["cells"] if cell["id"] == cell_id)
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "workshop_llm_agent_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _load_workflow(agent_module):
    spec = importlib.util.spec_from_file_location(
        "module3_interactive_workflow_test", WORKFLOW_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.modules.get("workshop_llm_agent")
    sys.modules["workshop_llm_agent"] = agent_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("workshop_llm_agent", None)
        else:
            sys.modules["workshop_llm_agent"] = previous
    return module


class _RecordingOutput(widgets.Output):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enter_count = 0
        self.rendered_errors = []

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self.rendered_errors.append(f"{exc_type.__name__}: {exc_value}")
        return True


def _widget_text(widget):
    text = []
    for attribute in ("description", "value"):
        value = getattr(widget, attribute, None)
        if isinstance(value, str):
            text.append(value)
    for child in getattr(widget, "children", ()):
        text.append(_widget_text(child))
    return "\n".join(text)


def _write_candidate_workspace(path, *, count=96):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    frame = pd.read_csv(SNAPSHOT_PATH).iloc[:count].copy()
    molecules = [Chem.MolFromSmiles(smile) for smile in frame["smile"]]
    assert all(molecule is not None for molecule in molecules)
    frame["MolWt"] = [Descriptors.MolWt(molecule) for molecule in molecules]
    frame["cLogP"] = [Descriptors.MolLogP(molecule) for molecule in molecules]
    frame["TPSA"] = [Descriptors.TPSA(molecule) for molecule in molecules]
    frame.to_csv(path / "reframe_candidates.csv", index=False)
    return frame


def _install_fake_panel_nvmolkit(monkeypatch, *, result_shape):
    import numpy as np
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    state = {"calls": [], "centroids": [], "first_members": []}

    class FakeAsyncResult:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return np.asarray(self.value)

    class FakeFingerprintBatch:
        def __init__(self, fingerprints):
            self.fingerprints = fingerprints

        def torch(self):
            return self

    class FakeMorganFingerprintGenerator:
        def __init__(self, *, radius, fpSize):
            self.generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=fpSize
            )

        def GetFingerprints(self, molecules, *, num_threads):
            assert num_threads == 0
            return FakeFingerprintBatch(
                list(self.generator.GetFingerprints(molecules, numThreads=0))
            )

    def cross_tanimoto_similarity(left, right):
        return FakeAsyncResult(
            [
                DataStructs.BulkTanimotoSimilarity(query, right.fingerprints)
                for query in left.fingerprints
            ]
        )

    def fused_butina(fingerprint_tensor, *, cutoff, return_centroids):
        state["calls"].append((cutoff, return_centroids))
        distances = []
        for row_index in range(1, len(fingerprint_tensor.fingerprints)):
            distances.extend(
                DataStructs.BulkTanimotoSimilarity(
                    fingerprint_tensor.fingerprints[row_index],
                    fingerprint_tensor.fingerprints[:row_index],
                    returnDistance=True,
                )
            )
        clusters = tuple(
            tuple(int(member) for member in cluster)
            for cluster in Butina.ClusterData(
                distances,
                len(fingerprint_tensor.fingerprints),
                cutoff,
                isDistData=True,
                reordering=True,
            )
        )
        centroids = np.asarray([cluster[-1] for cluster in clusters], dtype=int)
        state["centroids"] = centroids.tolist()
        state["first_members"] = [cluster[0] for cluster in clusters]
        if result_shape == "v05":
            cumulative_sizes = np.concatenate(
                (
                    np.zeros(1, dtype=int),
                    np.cumsum(
                        np.asarray([len(cluster) for cluster in clusters], dtype=int)
                    ),
                )
            )
            return (
                tuple(FakeAsyncResult(cluster) for cluster in clusters),
                FakeAsyncResult(cumulative_sizes),
                FakeAsyncResult(centroids),
            )
        labels = np.full(len(fingerprint_tensor.fingerprints), -1, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            labels[list(cluster)] = len(clusters) - cluster_id - 1
        reversed_centroids = centroids[::-1]
        if result_shape == "v06":
            return FakeAsyncResult(labels), FakeAsyncResult(reversed_centroids)
        if result_shape == "malformed":
            return FakeAsyncResult(labels[:-1]), FakeAsyncResult(reversed_centroids)
        raise AssertionError(f"unexpected fake result shape: {result_shape}")

    fake_torch = ModuleType("torch")
    fake_torch.Tensor = type("FakeTensor", (), {})
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_nvmolkit = ModuleType("nvmolkit")
    fake_nvmolkit.__path__ = []
    fake_clustering = ModuleType("nvmolkit.clustering")
    fake_clustering.fused_butina = fused_butina
    fake_fingerprints = ModuleType("nvmolkit.fingerprints")
    fake_fingerprints.MorganFingerprintGenerator = FakeMorganFingerprintGenerator
    fake_similarity = ModuleType("nvmolkit.similarity")
    fake_similarity.crossTanimotoSimilarity = cross_tanimoto_similarity
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "nvmolkit", fake_nvmolkit)
    monkeypatch.setitem(sys.modules, "nvmolkit.clustering", fake_clustering)
    monkeypatch.setitem(sys.modules, "nvmolkit.fingerprints", fake_fingerprints)
    monkeypatch.setitem(sys.modules, "nvmolkit.similarity", fake_similarity)
    return state


def _execute_panel_source_in_process(arguments, **kwargs):
    assert arguments[1:] == ["-I", "-"]
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_cwd = Path.cwd()
    return_code = 0
    try:
        os.chdir(kwargs["cwd"])
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(
                compile(kwargs["input"], "<isolated-panel-analysis>", "exec"),
                {"__name__": "__main__"},
            )
    except Exception:
        return_code = 1
        traceback.print_exc(file=stderr)
    finally:
        os.chdir(previous_cwd)
    return SimpleNamespace(
        returncode=return_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _panel_plan_payload():
    return {
        "data_observations": [
            "The fixed candidate pool contains 96 unique connectivity keys.",
            "Molecular weight, cLogP, and TPSA define the bounded coverage audit.",
        ],
        "strategies": [
            {
                "title": "Cluster-first coverage",
                "approach": "Select deterministic Butina representatives, then retain descriptor-range coverage.",
                "property_coverage_measure": "Mean normalized MolWt, cLogP, and TPSA ranges.",
                "cluster_balance": "Prefer separated representatives across the fixed pool.",
                "tradeoff": "Cluster representatives depend on fingerprint parameters.",
            },
            {
                "title": "Greedy max-min coverage",
                "approach": "Seed descriptor extrema, then add deterministic farthest fingerprint points.",
                "property_coverage_measure": "Mean normalized MolWt, cLogP, and TPSA ranges.",
                "cluster_balance": "Favor separation while retaining the measured descriptor range.",
                "tradeoff": "Farthest-point selection can favor unusual structural features.",
            },
        ],
        "recommended_strategy": 2,
        "recommendation_reason": (
            "The fixed candidate set supports a direct audited comparison with the "
            "first 24 source rows."
        ),
    }


def _panel_audit_payload():
    return {
        "result_assessment": (
            "The validated panel meets the fixed structural-distance and descriptor-coverage contract."
        ),
        "surprising_result": "Descriptor extrema can improve coverage without reducing the baseline distance.",
        "scientific_boundaries": (
            "These fingerprint and descriptor results do not establish biological activity, safety, or efficacy."
        ),
        "next_iteration": (
            "Test representation sensitivity and add assay-specific constraints before experimental use."
        ),
    }


class _Completions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        call = SimpleNamespace(
            function=SimpleNamespace(
                name="submit_neighborhood_policy",
                arguments=json.dumps(self.payload),
            )
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))]
        )


def _client_for(payload):
    completions = _Completions(payload)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _clear_workshop_key_environment(monkeypatch):
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_workshop_uses_verified_inference_hub_endpoint_and_model():
    agent = _load_agent()

    assert agent.NVIDIA_BASE_URL == "https://inference-api.nvidia.com/v1"
    assert agent.DEFAULT_MODEL == "nvidia/nvidia/nemotron-3-nano-30b-a3b"


def test_workshop_key_loads_protected_launch_file_without_prompt(monkeypatch, tmp_path):
    agent = _load_agent()
    saved_key = "sk-" + "saved" * 5
    key_path = tmp_path / "NVIDIA_INFERENCE_API_KEY"
    key_path.write_text(saved_key, encoding="utf-8")
    key_path.chmod(0o600)
    _clear_workshop_key_environment(monkeypatch)
    monkeypatch.setattr(
        agent, "_NVIDIA_INFERENCE_API_KEY_PATH", key_path, raising=False
    )
    monkeypatch.setattr(
        agent.getpass,
        "getpass",
        lambda prompt: pytest.fail("a valid protected key must not prompt"),
    )

    assert agent.get_workshop_api_key() == saved_key


def test_workshop_key_precedence_does_not_inspect_protected_file(monkeypatch, tmp_path):
    agent = _load_agent()
    unsafe_path = tmp_path / "NVIDIA_INFERENCE_API_KEY"
    unsafe_path.write_text("sk-unsafe-file", encoding="utf-8")
    unsafe_path.chmod(0o644)
    monkeypatch.setattr(
        agent, "_NVIDIA_INFERENCE_API_KEY_PATH", unsafe_path, raising=False
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-legacy-environment-key")
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "sk-primary-environment-key")

    assert agent.get_workshop_api_key("sk-explicit-key") == "sk-explicit-key"
    assert agent.get_workshop_api_key() == "sk-primary-environment-key"


def test_workshop_key_uses_legacy_variable_name_for_inference_hub_key(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _clear_workshop_key_environment(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-legacy-environment-key")
    monkeypatch.setattr(
        agent,
        "_NVIDIA_INFERENCE_API_KEY_PATH",
        tmp_path / "missing" / "NVIDIA_INFERENCE_API_KEY",
        raising=False,
    )

    assert agent.get_workshop_api_key(prompt=False) == "sk-legacy-environment-key"


def test_invalid_primary_key_fails_closed_without_using_legacy_or_saved_key(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    key_path = tmp_path / "NVIDIA_INFERENCE_API_KEY"
    key_path.write_text("sk-saved-key", encoding="utf-8")
    key_path.chmod(0o600)
    monkeypatch.setattr(
        agent, "_NVIDIA_INFERENCE_API_KEY_PATH", key_path, raising=False
    )
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "nvapi-build-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-legacy-key")
    monkeypatch.setattr(
        agent.getpass,
        "getpass",
        lambda prompt: pytest.fail("an invalid primary key must fail closed"),
    )

    with pytest.raises(ValueError, match="Inference Hub") as captured:
        agent.get_workshop_api_key()

    assert "sk-" in str(captured.value)
    assert "nvapi-" in str(captured.value)


@pytest.mark.parametrize("unsafe_kind", ["mode", "symlink", "oversize"])
def test_workshop_key_rejects_unsafe_protected_file_without_prompt(
    monkeypatch, tmp_path, unsafe_kind
):
    agent = _load_agent()
    secret = "sk-" + "do-not-leak" * 400
    target = tmp_path / "target"
    target.write_text(secret, encoding="utf-8")
    target.chmod(0o600)
    key_path = tmp_path / "NVIDIA_INFERENCE_API_KEY"
    if unsafe_kind == "mode":
        key_path.write_text(secret, encoding="utf-8")
        key_path.chmod(0o644)
    elif unsafe_kind == "symlink":
        key_path.symlink_to(target)
    else:
        key_path.write_text(secret, encoding="utf-8")
        key_path.chmod(0o600)
    _clear_workshop_key_environment(monkeypatch)
    monkeypatch.setattr(
        agent, "_NVIDIA_INFERENCE_API_KEY_PATH", key_path, raising=False
    )
    monkeypatch.setattr(
        agent.getpass,
        "getpass",
        lambda prompt: pytest.fail("an unsafe protected file must not prompt"),
    )

    with pytest.raises(ValueError) as captured:
        agent.get_workshop_api_key()

    assert secret not in str(captured.value)
    if unsafe_kind == "mode":
        assert "mode 0600" in str(captured.value)
    elif unsafe_kind == "symlink":
        assert "opened securely" in str(captured.value)
    else:
        assert "unexpectedly large" in str(captured.value)


def test_workshop_key_prompts_only_when_protected_file_is_missing(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    prompted_key = "sk-local-fallback"
    prompts = []
    _clear_workshop_key_environment(monkeypatch)
    monkeypatch.setattr(
        agent,
        "_NVIDIA_INFERENCE_API_KEY_PATH",
        tmp_path / "missing" / "NVIDIA_INFERENCE_API_KEY",
        raising=False,
    )
    monkeypatch.setattr(
        agent.getpass,
        "getpass",
        lambda prompt: prompts.append(prompt) or prompted_key,
    )

    assert agent.get_workshop_api_key(prompt=True) == prompted_key
    assert len(prompts) == 1
    with pytest.raises(ValueError, match="NVIDIA_INFERENCE_API_KEY"):
        agent.get_workshop_api_key(prompt=False)
    assert len(prompts) == 1


@pytest.mark.parametrize(
    "source",
    ["explicit", "primary_environment", "legacy_environment", "saved_file"],
)
def test_workshop_key_rejects_build_keys_with_actionable_guidance(
    monkeypatch, tmp_path, source
):
    agent = _load_agent()
    _clear_workshop_key_environment(monkeypatch)
    key_path = tmp_path / "NVIDIA_INFERENCE_API_KEY"
    monkeypatch.setattr(
        agent, "_NVIDIA_INFERENCE_API_KEY_PATH", key_path, raising=False
    )
    explicit_key = None
    if source == "explicit":
        explicit_key = "nvapi-explicit-build-key"
    elif source == "primary_environment":
        monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "nvapi-primary-build-key")
    elif source == "legacy_environment":
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-legacy-build-key")
    else:
        key_path.write_text("nvapi-saved-build-key", encoding="utf-8")
        key_path.chmod(0o600)

    with pytest.raises(ValueError, match="Inference Hub") as captured:
        agent.get_workshop_api_key(explicit_key, prompt=False)

    assert "sk-" in str(captured.value)
    assert "nvapi-" in str(captured.value)


def test_workshop_mode_defaults_to_interactive_and_accepts_only_two_modes(monkeypatch):
    agent = _load_agent()

    monkeypatch.delenv("NVMOLKIT_WORKSHOP_MODE", raising=False)
    assert agent.workshop_mode() == "interactive"
    monkeypatch.setenv("NVMOLKIT_WORKSHOP_MODE", " reference ")
    assert agent.workshop_mode() == "reference"
    monkeypatch.setenv("NVMOLKIT_WORKSHOP_MODE", "other")
    with pytest.raises(ValueError, match="interactive or reference"):
        agent.workshop_mode()


def test_reference_mode_uses_labeled_fixed_policy_without_key_or_client(monkeypatch):
    agent = _load_agent()

    def forbidden_key(*args, **kwargs):
        raise AssertionError("reference mode requested a key")

    monkeypatch.setattr(agent, "get_workshop_api_key", forbidden_key)
    implementation = agent.select_neighborhood_implementation(
        "bounded neighborhood lesson", mode="reference", client=object()
    )

    assert implementation.label == "reference"
    assert implementation.policy.missing_anchor == "raise"
    assert implementation.policy.invalid_matrix == "raise"
    assert "build_neighborhood_atlas" in implementation.function_source


def test_interactive_mode_uses_exact_policy_schema_and_python_owned_renderer(
    monkeypatch,
):
    agent = _load_agent()
    payload = {
        "MISSING_ANCHOR": "skip",
        "INVALID_MATRIX": "raise",
        "MISSING_ANCHOR_EXPLANATION": "Continue with anchors present in the fixed sample.",
        "INVALID_MATRIX_EXPLANATION": "Stop if the aligned similarity matrix is invalid.",
    }
    client, completions = _client_for(payload)
    key_requests = []

    def protected_key(*args, **kwargs):
        key_requests.append((args, kwargs))
        return "protected-key"

    monkeypatch.setattr(agent, "get_workshop_api_key", protected_key)
    implementation = agent.select_neighborhood_implementation(
        "bounded neighborhood lesson", mode="interactive", client=client
    )

    assert implementation.label == "hosted_nemotron"
    assert implementation.policy.missing_anchor == "skip"
    assert implementation.policy.invalid_matrix == "raise"
    assert key_requests
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "nvidia/nvidia/nemotron-3-nano-30b-a3b"
    assert call["parallel_tool_calls"] is False
    assert set(call["tools"][0]["function"]["parameters"]["properties"]) == {
        "MISSING_ANCHOR",
        "INVALID_MATRIX",
        "MISSING_ANCHOR_EXPLANATION",
        "INVALID_MATRIX_EXPLANATION",
    }
    assert payload["MISSING_ANCHOR_EXPLANATION"] not in implementation.function_source
    assert "if matches.empty:\n            continue" in implementation.function_source


def test_interactive_mode_redacts_key_shaped_provider_failures(monkeypatch):
    agent = _load_agent()
    key = "nvapi-" + "x" * 24

    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError(f"provider rejected {key}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    monkeypatch.setattr(agent, "get_workshop_api_key", lambda *args, **kwargs: key)

    with pytest.raises(agent.WorkshopAgentError) as captured:
        agent.select_neighborhood_implementation(
            "bounded neighborhood lesson", mode="interactive", client=client
        )

    assert key not in str(captured.value)


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("provider returned sk-abc.def/ghi+jkl=", "sk-abc.def/ghi+jkl="),
        ("provider returned nvapi-abc.def/ghi+jkl=", "nvapi-abc.def/ghi+jkl="),
        (
            'request={"NVIDIA_INFERENCE_API_KEY": "sk-json-secret"}',
            "sk-json-secret",
        ),
        ("request={'NVIDIA_API_KEY': 'sk-dict-secret'}", "sk-dict-secret"),
        ("NVIDIA_INFERENCE_API_KEY=opaque-secret", "opaque-secret"),
    ],
)
def test_sensitive_text_redacts_both_key_names_and_key_shapes(raw, secret):
    agent = _load_agent()

    redacted = agent._redact_sensitive_text(raw)

    assert secret not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.parametrize(
    "safe_text",
    [
        "task-id remains visible",
        "mask-id remains visible",
        "risk-id remains visible",
    ],
)
def test_sensitive_text_does_not_redact_sk_substrings_in_ordinary_words(safe_text):
    agent = _load_agent()

    assert agent._redact_sensitive_text(safe_text) == safe_text


@pytest.mark.parametrize("malformed", [False, True])
def test_interactive_failures_do_not_expose_key_in_visible_traceback(
    monkeypatch, malformed
):
    agent = _load_agent()
    key = "nvapi-" + "s" * 24

    if malformed:
        client, _ = _client_for(
            {
                "MISSING_ANCHOR": "raise",
                "INVALID_MATRIX": "raise",
                "MISSING_ANCHOR_EXPLANATION": key,
            }
        )
    else:

        class FailingCompletions:
            def create(self, **kwargs):
                raise RuntimeError(f"provider raw failure: {key}")

        client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    monkeypatch.setattr(agent, "get_workshop_api_key", lambda *args, **kwargs: key)

    with pytest.raises(agent.WorkshopAgentError) as captured:
        agent.select_neighborhood_implementation(
            "bounded neighborhood lesson", mode="interactive", client=client
        )

    visible = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert key not in str(captured.value)
    assert key not in visible
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize("direct", [True, False])
def test_structured_request_does_not_retain_raw_provider_context(monkeypatch, direct):
    agent = _load_agent()
    key = "nvapi-" + "z" * 24

    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError(key)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    monkeypatch.setattr(agent, "get_workshop_api_key", lambda *args, **kwargs: key)
    with pytest.raises(agent.WorkshopAgentError) as captured:
        if direct:
            agent._structured_request(
                api_key=key,
                system_prompt="policy",
                user_prompt="policy",
                tool_name="submit_neighborhood_policy",
                response_model=agent.NeighborhoodPolicy,
                max_tokens=1,
                client=client,
            )
        else:
            agent.select_neighborhood_implementation(
                "policy", mode="interactive", client=client
            )
    visible = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert key not in visible
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_module2_sources_have_no_stale_paths_or_generated_code_ingestion():
    agent_source = MODULE_PATH.read_text(encoding="utf-8")
    notebook_source = NOTEBOOK_PATH.read_text(encoding="utf-8")

    for source in (agent_source, notebook_source):
        assert "/nvmolkit-brev-notebook" not in source
        assert "/.venv" not in source
    for obsolete_name in (
        "GeneratedFunction",
        "generate_neighborhood_function",
        "validate_neighborhood_function_source",
        "_strip_code_fence",
        "_repair_missing_function_body_indent",
        "_compact_neighborhood_source",
        "_ensure_panel_imports",
    ):
        assert obsolete_name not in agent_source


def test_module2_legacy_source_ingestion_is_physically_absent():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "# ruff: noqa: F821" not in source
    assert not {name for name in definitions if name.startswith("_retired_")}
    assert not {
        "generate_neighborhood_function",
        "review_neighborhood_function",
        "validate_neighborhood_function_source",
        "_strip_code_fence",
        "_repair_missing_function_body_indent",
        "_compact_neighborhood_source",
    }.intersection(definitions)
    assert "proposes Python source" not in source
    assert "Do not return code, JSON, or Markdown fences" not in source


def test_module2_notebook_uses_plain_policy_explanations_and_consistent_flow():
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert 'display(Markdown("**Policy explanations**' not in source
    assert 'print(\\"Missing anchor explanation:\\"' in source
    assert 'print(\\"Invalid matrix explanation:\\"' in source
    assert "<img src=" not in source
    for stale_text in (
        "untrusted generated code",
        "copy only the complete",
        "starter function",
        "pasted function",
        "sends the implementation",
    ):
        assert stale_text not in source.lower()
    for required_text in (
        "only the bounded policy prompt",
        "does not send dataset rows, rendered code, or analysis results",
        "Hosted mode asks Nemotron to choose the two bounded policy values for this run",
        "Reference mode uses fixed local reference policy values with no hosted selection",
        "In both modes, Python applies the matching allow-listed implementation",
        "You evaluate the choices afterward, run the checks, and interpret the result",
        "## Step 3 — Validate and bind the locally rendered function",
        "## Step 4 — Run the bound function and its normal-path invariant checks",
        "Selected failure branches were not triggered",
        "## Step 6 — Review the local receipt and results",
    ):
        assert required_text in source


def test_module2_notebook_has_one_recoverable_bounded_attendee_flow():
    source = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    for required_text in (
        "hosted nemotron returns only two bounded policy choices plus explanations",
        "python renders, validates, and binds the function",
        "hosted mode asks nemotron to choose the two bounded policy values for this run",
        "reference mode uses fixed local reference policy values with no hosted selection",
        "in both modes, python applies the matching allow-listed implementation",
        "you evaluate the choices afterward, run the checks, and interpret the result",
        "set `nvmolkit_workshop_mode=reference`, restart, and rerun the notebook",
        "zero client calls",
        "## step 4 — run the bound function and its normal-path invariant checks",
        "selected failure branches were not triggered",
        "selected failure policies, representation sensitivity, and scientific interpretation",
    ):
        assert required_text in source
    for stale_text in (
        "returned source",
        "generation enabled",
        "embedded agent disabled",
        "participant implementation",
        "reference_build_neighborhood_atlas",
        "function you defined",
        "agent looped per molecule",
        "writing the function",
    ):
        assert stale_text not in source


def test_module2_discussion_and_answer_key_pair_the_three_semantic_items():
    discussion = _notebook_cell_source(NOTEBOOK_PATH, "cell-17376c167229")
    answer_key = _notebook_cell_source(NOTEBOOK_PATH, "2db02500")
    normalized_discussion = " ".join(discussion.split())
    assert (
        "Are both selected policies appropriate? If not, which values would you choose "
        "and why?"
    ) in normalized_discussion
    assert "radius-sensitive anchor" in discussion.lower()
    assert "unsupported scientific inference" in discussion.lower()
    assert "`raise`/`raise` is appropriate for the fixed teaching run" in answer_key
    assert "recorded run values remain the values actually used" in answer_key
    assert "recorded Nemotron values" not in answer_key
    assert "selected `skip` would continue" in answer_key
    assert "lowest top-10 Jaccard overlap" in answer_key
    assert "binding, activity, ADMET, efficacy, or safety" in answer_key


def test_module2_roles_distinguish_hosted_selection_from_local_reference_values():
    goal = _notebook_cell_source(NOTEBOOK_PATH, "cell-65b035d14880")
    receipt = _notebook_cell_source(NOTEBOOK_PATH, "embedded-agent-review")
    answer_key = _notebook_cell_source(NOTEBOOK_PATH, "2db02500")
    plan = PLAN_PATH.read_text(encoding="utf-8")
    design = SPEC_PATH.read_text(encoding="utf-8")

    for source in (goal, plan, design):
        normalized_source = " ".join(source.split())
        assert (
            "Hosted mode asks Nemotron to choose the two bounded policy values for this run"
            in normalized_source
        )
        assert (
            "Reference mode uses fixed local reference policy values with no hosted "
            "selection" in normalized_source
        )
    assert 'if implementation.label == "hosted_nemotron":' in receipt
    assert 'elif implementation.label == "reference":' in receipt
    assert "Hosted mode: Nemotron chose the two bounded policy values" in receipt
    assert "fixed local reference policy values were used" in receipt
    assert "no hosted selection occurred" in receipt
    assert "hosted receipt" not in receipt.lower()
    assert "policy receipt" in receipt.lower()
    assert "Python applies" in receipt
    assert "You evaluate" in receipt
    assert "recorded run values remain the values actually used" in answer_key
    assert "recorded Nemotron values" not in answer_key
    assert "recorded run values remain the values actually used" in plan
    assert "recorded Nemotron values" not in plan
    assert "you select and assess failure policies" not in goal.lower()


def test_module2_validation_copy_is_limited_to_normal_path_invariants():
    notebook_source = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    check_cell = _notebook_cell_source(NOTEBOOK_PATH, "cell-b766b437bdb5")

    assert "all acceptance tests" not in notebook_source
    assert "normal-path invariant checks" in notebook_source
    assert "selected failure branches were not triggered" in notebook_source
    assert (
        'attendee_columns = ["radius", "query", "rank", "neighbor", "tanimoto"]'
        in check_cell
    )
    assert "display(attendee_atlas.head(12))" in check_cell


def test_panel_metric_definitions_match_the_approved_contract():
    import pandas as pd

    agent = _load_agent()
    similarities = [
        [1.0, 0.25, 0.80],
        [0.25, 1.0, 0.40],
        [0.80, 0.40, 1.0],
    ]
    candidates = pd.DataFrame(
        {
            "MolWt": [10.0, 20.0, 30.0],
            "cLogP": [2.0, 2.0, 2.0],
            "TPSA": [0.0, 50.0, 100.0],
        }
    )
    selected = candidates.iloc[[0, 2]]

    assert agent.minimum_pairwise_distance(similarities) == pytest.approx(0.20)
    assert agent.descriptor_range_coverage(candidates, selected) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", [None, "interactive", "Reference", True, 1])
def test_panel_agent_accepts_only_exact_hosted_or_reference_modes(tmp_path, mode):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)

    with pytest.raises(ValueError, match="mode must be 'hosted' or 'reference'"):
        agent.PanelDesignAgent(workdir=tmp_path, mission="bounded mission", mode=mode)


@pytest.mark.parametrize(
    ("api_key", "client"),
    [("nvapi-" + "x" * 24, None), (None, object())],
)
def test_panel_reference_mode_rejects_any_key_or_client(tmp_path, api_key, client):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)

    with pytest.raises(
        ValueError, match="Reference mode requires no API key or client"
    ):
        agent.PanelDesignAgent(
            workdir=tmp_path,
            mission="bounded mission",
            mode="reference",
            api_key=api_key,
            client=client,
        )


def test_panel_reference_mode_is_key_free_and_makes_zero_client_calls(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-" + "ambient-secret" * 2)
    monkeypatch.setattr(
        agent,
        "get_workshop_api_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference mode requested a hosted key")
        ),
    )
    monkeypatch.setattr(
        agent,
        "_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference mode created a hosted client")
        ),
    )

    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    plan = panel_agent.request_plan()

    assert isinstance(plan, agent.PanelPlan)
    assert len(plan.strategies) == 2
    assert panel_agent.mode == "reference"
    assert panel_agent.api_key is None
    assert panel_agent.client is None


def test_panel_agent_requires_exact_fixed_candidate_inventory(tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path, count=95)

    with pytest.raises(
        agent.WorkshopAgentError,
        match="exactly 96 unique candidate connectivity keys",
    ):
        agent.PanelDesignAgent(
            workdir=tmp_path,
            mission="bounded mission",
            mode="reference",
        )


def test_hosted_panel_agent_calls_only_strict_plan_and_audit_schemas(tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    calls = []

    class PanelCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            name = kwargs["tool_choice"]["function"]["name"]
            payload = (
                _panel_plan_payload()
                if name == "submit_panel_plan"
                else _panel_audit_payload()
            )
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name=name, arguments=json.dumps(payload))
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=PanelCompletions()))
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="hosted",
        api_key="sk-" + "x" * 24,
        client=client,
    )
    panel_agent.request_plan()
    report_snapshot = json.dumps(
        {
            "candidate_count": 96,
            "panel_count": 24,
            "acceptance": {
                "baseline_minimum_distance": 0.2,
                "selected_minimum_distance": 0.3,
                "baseline_descriptor_coverage": 0.4,
                "selected_descriptor_coverage": 0.8,
                "passed": True,
            },
        }
    )
    audit = panel_agent._request_audit(2, report_snapshot)

    assert isinstance(audit, agent.PanelAudit)
    assert [call["tool_choice"]["function"]["name"] for call in calls] == [
        "submit_panel_plan",
        "submit_panel_audit",
    ]
    for call in calls:
        function = call["tools"][0]["function"]
        assert function["strict"] is True
        assert call["parallel_tool_calls"] is False
        assert function["parameters"]["additionalProperties"] is False


def test_module3_has_no_model_source_schema_or_ingestion_path():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for removed in (
        "GeneratedAnalysis",
        "submit_panel_analysis",
        "_request_analysis",
        "_generation_prompt",
        "generated_source",
    ):
        assert removed not in source
        assert removed not in definitions


@pytest.mark.parametrize("approved_strategy", [1, 2])
def test_panel_source_is_exactly_controller_rendered(approved_strategy):
    agent = _load_agent()
    source = agent._render_panel_analysis(approved_strategy, 24)

    assert (
        agent.validate_panel_analysis_source(
            source,
            approved_strategy=approved_strategy,
            expected_panel_size=24,
        )
        == source
    )
    with pytest.raises(
        agent.WorkshopAgentError,
        match="exact controller-rendered implementation",
    ):
        agent.validate_panel_analysis_source(
            source + "\n# hosted or local mutation\n",
            approved_strategy=approved_strategy,
            expected_panel_size=24,
        )


def test_panel_source_embeds_and_uses_the_exact_fused_result_normalizer():
    agent = _load_agent()
    source = agent._render_panel_analysis(1, 24)
    normalizer_source = textwrap.dedent(
        inspect.getsource(agent.normalize_fused_butina_result)
    )
    tree = ast.parse(source)

    assert source.count(normalizer_source) == 1
    compile(source, "<rendered-panel-analysis>", "exec")
    assert "from nvmolkit_compat import" not in source
    assert "from notebooks.nvmolkit_compat import" not in source
    fused_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fused_butina"
    ]
    assert len(fused_calls) == 1
    fused_keywords = {keyword.arg: keyword.value for keyword in fused_calls[0].keywords}
    assert isinstance(fused_keywords["cutoff"], ast.Name)
    assert fused_keywords["cutoff"].id == "DISTANCE_CUTOFF"
    assert ast.literal_eval(fused_keywords["return_centroids"]) is True
    assignments = {
        tuple(element.id for element in node.targets[0].elts): node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and all(isinstance(element, ast.Name) for element in node.targets[0].elts)
    }
    normalized_call = assignments[("cluster_labels", "clusters", "centroid_indices")]
    assert isinstance(normalized_call, ast.Call)
    assert isinstance(normalized_call.func, ast.Name)
    assert normalized_call.func.id == "normalize_fused_butina_result"
    assert isinstance(normalized_call.args[0], ast.Name)
    assert normalized_call.args[0].id == "raw_result"
    assert len(normalized_call.keywords) == 1
    assert normalized_call.keywords[0].arg == "molecule_count"
    assert isinstance(normalized_call.keywords[0].value, ast.Name)
    assert normalized_call.keywords[0].value.id == "CANDIDATE_COUNT"
    fused_assignment = source.index("raw_result = fused_butina(")
    fused_branch_start = source.rindex("if nvmolkit_ready:", 0, fused_assignment)
    fused_branch_end = source.index("else:", fused_assignment)
    fused_branch = source[fused_branch_start:fused_branch_end]
    assert (
        "centroid_indices = [int(index) for index in centroid_indices]" in fused_branch
    )
    assert "member_indices[0]" not in fused_branch
    assert "centroid_indices.append" not in fused_branch


def test_panel_v05_and_v06_results_keep_the_same_validated_partition(
    monkeypatch, tmp_path
):
    import pandas as pd

    agent = _load_agent()
    results = {}
    for result_shape in ("v05", "v06"):
        workdir = tmp_path / result_shape
        workdir.mkdir()
        _write_candidate_workspace(workdir)
        state = _install_fake_panel_nvmolkit(monkeypatch, result_shape=result_shape)
        panel_agent = agent.PanelDesignAgent(
            workdir=workdir,
            mission="bounded mission",
            mode="reference",
        )
        panel_agent.request_plan()
        monkeypatch.setattr(agent.subprocess, "run", _execute_panel_source_in_process)

        run = panel_agent.run(
            approved_strategy=1,
            expected_panel_size=24,
            timeout_seconds=120,
        )

        assert run.success, run.attempts[0].stderr_tail
        assert state["calls"] == [(0.55, True)]
        assert any(
            centroid != first_member
            for centroid, first_member in zip(
                state["centroids"], state["first_members"], strict=True
            )
        )
        panel = pd.read_csv(run.panel_path)
        report = json.loads(run.report_path.read_text(encoding="utf-8"))
        candidates = pd.read_csv(workdir / "reframe_candidates.csv")
        index_by_key = {
            key: index for index, key in enumerate(candidates["canonical_ikey"])
        }
        selected_indices = [
            index_by_key[key] for key in panel["canonical_ikey"].tolist()
        ]
        descriptor_extrema = {
            int(index)
            for column in ("MolWt", "cLogP", "TPSA")
            for index in (candidates[column].idxmin(), candidates[column].idxmax())
        }
        selected_after_seeding = set(selected_indices) - descriptor_extrema
        assert len(panel) == 24
        assert panel["canonical_ikey"].nunique() == 24
        assert panel["selection_order"].tolist() == list(range(1, 25))
        assert report["acceptance"]["passed"] is True
        assert selected_after_seeding <= set(state["centroids"])
        results[result_shape] = (panel, report)

    old_panel, old_report = results["v05"]
    new_panel, new_report = results["v06"]
    assert old_panel["canonical_ikey"].tolist() == new_panel["canonical_ikey"].tolist()
    old_labels = old_panel["method_cluster"].tolist()
    new_labels = new_panel["method_cluster"].tolist()
    assert old_labels != new_labels
    assert all(
        (old_labels[left] == old_labels[right])
        == (new_labels[left] == new_labels[right])
        for left in range(len(old_labels))
        for right in range(left + 1, len(old_labels))
    )
    assert old_report["cluster_coverage"] == new_report["cluster_coverage"]


def test_panel_malformed_fused_result_fails_before_result_artifacts(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    (tmp_path / "panel.csv").write_text("stale panel", encoding="utf-8")
    (tmp_path / "report.json").write_text("stale report", encoding="utf-8")
    (tmp_path / "agent_trace.json").write_text(
        json.dumps({"success": True}), encoding="utf-8"
    )
    state = _install_fake_panel_nvmolkit(monkeypatch, result_shape="malformed")
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    monkeypatch.setattr(agent.subprocess, "run", _execute_panel_source_in_process)

    run = panel_agent.run(
        approved_strategy=1,
        expected_panel_size=24,
        timeout_seconds=120,
    )

    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))
    assert state["calls"] == [(0.55, True)]
    assert run.success is False
    assert "Malformed fused Butina result." in run.attempts[0].stderr_tail
    assert not run.panel_path.exists()
    assert not run.report_path.exists()
    assert trace["success"] is False
    assert trace["attempts"][0]["passed"] is False


def test_panel_child_process_does_not_receive_hosted_key(monkeypatch, tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    child_environments = []

    def fake_run(*args, **kwargs):
        child_environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setenv("NVIDIA_API_KEY", "sk-" + "legacy-secret" * 3)
    monkeypatch.setenv(
        "NVIDIA_INFERENCE_API_KEY", "sk-" + "primary-secret" * 3
    )
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    monkeypatch.setattr(
        agent,
        "_validate_panel_artifacts_snapshot",
        lambda *args, **kwargs: (
            {
                "candidate_count": 96,
                "panel_count": 24,
                "acceptance_passed": True,
            },
            "{}",
        ),
    )
    monkeypatch.setattr(
        panel_agent, "_request_audit", lambda strategy, report_snapshot: None
    )

    run = panel_agent.run(approved_strategy=2, expected_panel_size=24)

    assert run.success
    assert len(child_environments) == 1
    assert "NVIDIA_API_KEY" not in child_environments[0]
    assert "NVIDIA_INFERENCE_API_KEY" not in child_environments[0]


def test_panel_executes_validated_source_after_callback_mutation_and_redacts_output(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    expected_source = agent._render_panel_analysis(2, 24)
    secret = "nvapi-" + "callback-file-secret" * 2
    executed_sources = []
    progress_events = []

    def mutate_public_artifact(event, payload):
        if event == "source_validated":
            (tmp_path / "analysis.py").write_text(
                f"print({secret!r})\n", encoding="utf-8"
            )
        progress_events.append((event, payload))

    def record_execution(arguments, **kwargs):
        if kwargs.get("input") is not None:
            executed_source = kwargs["input"]
        else:
            executed_path = Path(kwargs["cwd"]) / arguments[-1]
            executed_source = executed_path.read_text(encoding="utf-8")
        executed_sources.append(executed_source)
        return SimpleNamespace(
            returncode=0,
            stdout=f"untrusted output {secret}",
            stderr=f"NVIDIA_API_KEY={secret}",
        )

    monkeypatch.setattr(agent.subprocess, "run", record_execution)
    monkeypatch.setattr(
        agent,
        "_validate_panel_artifacts_snapshot",
        lambda *args, **kwargs: (
            {
                "candidate_count": 96,
                "panel_count": 24,
                "acceptance_passed": True,
            },
            "{}",
        ),
    )
    monkeypatch.setattr(
        panel_agent, "_request_audit", lambda strategy, report_snapshot: None
    )

    run = panel_agent.run(
        approved_strategy=2,
        expected_panel_size=24,
        progress_callback=mutate_public_artifact,
    )

    assert run.success
    assert executed_sources == [expected_source]
    assert run.analysis_path.read_text(encoding="utf-8") == expected_source
    trace_text = run.trace_path.read_text(encoding="utf-8")
    visible = repr(run.attempts) + repr(progress_events) + trace_text
    assert secret not in visible
    assert "[REDACTED]" in visible


def test_panel_rejects_candidate_mutation_after_preexecution_callback(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()

    def mutate_candidate(event, payload):
        if event == "execution_started":
            candidate_path = tmp_path / "reframe_candidates.csv"
            source = candidate_path.read_text(encoding="utf-8")
            candidate_path.write_text(source + "\n", encoding="utf-8")

    called = []

    def forbidden_child(*args, **kwargs):
        called.append(True)
        return SimpleNamespace(returncode=1, stdout="", stderr="must not run")

    monkeypatch.setattr(agent.subprocess, "run", forbidden_child)

    with pytest.raises(agent.WorkshopAgentError, match="candidate input changed"):
        panel_agent.run(
            approved_strategy=2,
            expected_panel_size=24,
            progress_callback=mutate_candidate,
        )

    assert called == []


def test_panel_preexecution_cleanup_never_follows_output_symlink(monkeypatch, tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    protected_path = tmp_path / "protected-key"
    protected_value = "nvapi-protected-file-must-not-change"
    protected_path.write_text(protected_value, encoding="utf-8")

    def install_output_symlink(event, payload):
        if event == "execution_started":
            (tmp_path / "panel.csv").symlink_to(protected_path)

    called = []

    def unsafe_child(*args, **kwargs):
        called.append(True)
        (tmp_path / "panel.csv").write_text("overwritten", encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="", stderr="unsafe")

    monkeypatch.setattr(agent.subprocess, "run", unsafe_child)

    with pytest.raises(agent.WorkshopAgentError, match="output paths"):
        panel_agent.run(
            approved_strategy=2,
            expected_panel_size=24,
            progress_callback=install_output_symlink,
        )

    assert called == []
    assert protected_path.read_text(encoding="utf-8") == protected_value


def test_panel_child_isolated_from_pythonpath_sitecustomize_and_unrelated_secrets(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()

    hostile_pythonpath = tmp_path / "hostile-pythonpath"
    hostile_pythonpath.mkdir()
    sitecustomize_marker = tmp_path / "sitecustomize-ran.txt"
    sentinel_value = "workshop-secret-must-not-reach-analysis"
    (hostile_pythonpath / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        f"Path({str(sitecustomize_marker)!r}).write_text("
        "os.environ.get('WORKSHOP_SENTINEL_SECRET', 'missing'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    child_environment_path = tmp_path / "child-environment.txt"
    child_arguments_path = tmp_path / "child-arguments.txt"
    python_wrapper = tmp_path / "recording-python"
    python_wrapper.write_text(
        "#!/bin/sh\n"
        f"/usr/bin/env > {shlex.quote(str(child_environment_path))}\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(child_arguments_path))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o700)
    monkeypatch.setenv("PYTHONPATH", str(hostile_pythonpath))
    monkeypatch.setenv("WORKSHOP_SENTINEL_SECRET", sentinel_value)

    run = panel_agent.run(
        approved_strategy=2,
        expected_panel_size=24,
        python_executable=str(python_wrapper),
        timeout_seconds=120,
    )

    assert run.success, run.attempts[0].message
    assert child_arguments_path.read_text(encoding="utf-8").splitlines() == [
        "-I",
        "-",
    ]
    child_environment = child_environment_path.read_text(encoding="utf-8")
    assert "PYTHONPATH=" not in child_environment
    assert "WORKSHOP_SENTINEL_SECRET=" not in child_environment
    assert sentinel_value not in child_environment
    assert not sitecustomize_marker.exists()


def test_panel_audit_uses_validated_report_snapshot_before_progress_callback(
    tmp_path,
):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    calls = []

    class PanelCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            name = kwargs["tool_choice"]["function"]["name"]
            payload = (
                _panel_plan_payload()
                if name == "submit_panel_plan"
                else _panel_audit_payload()
            )
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name=name, arguments=json.dumps(payload))
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=PanelCompletions()))
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="hosted",
        api_key="sk-" + "x" * 24,
        client=client,
    )
    panel_agent.request_plan()
    replacement_marker = "callback-replaced-unvalidated-report"

    def replace_report(event, payload):
        if event == "attempt_passed":
            (tmp_path / "report.json").write_text(
                json.dumps({"marker": replacement_marker}), encoding="utf-8"
            )

    run = panel_agent.run(
        approved_strategy=2,
        expected_panel_size=24,
        timeout_seconds=120,
        progress_callback=replace_report,
    )

    assert run.success, run.attempts[0].message
    audit_call = next(
        call
        for call in calls
        if call["tool_choice"]["function"]["name"] == "submit_panel_audit"
    )
    audit_prompt = audit_call["messages"][1]["content"]
    assert '"candidate_count": 96' in audit_prompt
    assert '"panel_count": 24' in audit_prompt
    assert replacement_marker not in audit_prompt


def test_panel_artifacts_reject_symlinks_wrong_counts_malformed_and_stale_files(
    tmp_path,
):
    agent = _load_agent()
    frame = _write_candidate_workspace(tmp_path)
    panel = frame.iloc[:24].copy()
    panel["selection_reason"] = "test"
    panel["method_cluster"] = range(24)
    panel["selection_order"] = range(1, 25)
    panel.to_csv(tmp_path / "valid-panel.csv", index=False)
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "panel.csv").symlink_to(tmp_path / "valid-panel.csv")

    with pytest.raises(agent.WorkshopAgentError, match="regular files"):
        agent.validate_panel_artifacts(tmp_path, expected_panel_size=24)

    (tmp_path / "panel.csv").unlink()
    panel.iloc[:23].to_csv(tmp_path / "panel.csv", index=False)
    with pytest.raises(agent.WorkshopAgentError, match="23 rows; expected 24"):
        agent.validate_panel_artifacts(tmp_path, expected_panel_size=24)

    panel.to_csv(tmp_path / "panel.csv", index=False)
    (tmp_path / "report.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(agent.WorkshopAgentError, match="not valid JSON"):
        agent.validate_panel_artifacts(tmp_path, expected_panel_size=24)

    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    old_seconds = 1_600_000_000
    os.utime(tmp_path / "panel.csv", (old_seconds, old_seconds))
    os.utime(tmp_path / "report.json", (old_seconds, old_seconds))
    with pytest.raises(agent.WorkshopAgentError, match="current execution"):
        agent.validate_panel_artifacts(
            tmp_path,
            expected_panel_size=24,
            not_before_ns=1_700_000_000_000_000_000,
        )


@pytest.mark.parametrize(
    ("audit_payload", "expected_status"),
    [
        (_panel_audit_payload(), "Analysis validated; audit complete"),
        (None, "Analysis validated; audit unavailable"),
    ],
)
def test_module3_widget_captures_callback_and_truthfully_labels_audit_state(
    monkeypatch, tmp_path, audit_payload, expected_status
):
    agent = _load_agent()
    workflow_module = _load_workflow(agent)
    plan = agent.PanelPlan.model_validate(_panel_plan_payload())
    audit = (
        agent.PanelAudit.model_validate(audit_payload)
        if audit_payload is not None
        else None
    )
    run = agent.PanelAgentRun(
        success=True,
        approved_strategy=1,
        attempts=(),
        analysis_path=tmp_path / "analysis.py",
        panel_path=tmp_path / "panel.csv",
        report_path=tmp_path / "report.json",
        trace_path=tmp_path / "agent_trace.json",
        audit=audit,
    )

    class FakeAgent:
        def __init__(self):
            self.plan_calls = 0
            self.run_calls = []

        def request_plan(self):
            self.plan_calls += 1
            return plan

        def run(self, **kwargs):
            self.run_calls.append(kwargs["approved_strategy"])
            if audit is None:
                kwargs["progress_callback"](
                    "audit_failed", {"message": "optional audit unavailable"}
                )
            else:
                kwargs["progress_callback"](
                    "audit_completed", {"audit": audit.model_dump()}
                )
            return run

    fake_agent = FakeAgent()
    callback_calls = []

    monkeypatch.setattr(workflow_module.widgets, "Output", _RecordingOutput)
    monkeypatch.setattr(workflow_module, "ipython_display", lambda value: value)
    workflow = workflow_module.launch_interactive_panel_design(
        fake_agent,
        expected_panel_size=24,
        on_complete=callback_calls.append,
    )

    assert workflow.status == "awaiting_approval"
    assert workflow.agent_run is None
    workflow.strategy_control.value = 1
    workflow._approve_and_run(workflow.approve_button)

    assert workflow.result_output in workflow.root.children
    assert workflow.result_output.enter_count == 1
    assert fake_agent.plan_calls == 1
    assert fake_agent.run_calls == [1]
    assert callback_calls == [run]
    assert workflow.plan.recommended_strategy == 2
    assert workflow.agent_run.approved_strategy == 1

    visible_text = _widget_text(workflow.root)
    assert expected_status in visible_text
    assert expected_status in workflow.transcript_text
    if run.audit is None:
        assert "Agent workflow complete" not in visible_text

    workflow._approve_and_run(workflow.approve_button)
    assert fake_agent.plan_calls == 1
    assert fake_agent.run_calls == [1]
    assert callback_calls == [run]


def test_module3_widget_redacts_callback_error_without_changing_success(
    monkeypatch, tmp_path
):
    agent = _load_agent()
    workflow_module = _load_workflow(agent)
    plan = agent.PanelPlan.model_validate(_panel_plan_payload())
    run = agent.PanelAgentRun(
        success=True,
        approved_strategy=2,
        attempts=(),
        analysis_path=tmp_path / "analysis.py",
        panel_path=tmp_path / "panel.csv",
        report_path=tmp_path / "report.json",
        trace_path=tmp_path / "agent_trace.json",
        audit=None,
    )

    class FakeAgent:
        def request_plan(self):
            return plan

        def run(self, **kwargs):
            return run

    secret = "nvapi-abc.def/ghi+jkl="
    named_secret = "named.secret/value+tail="
    callback_calls = []

    def failing_callback(result):
        callback_calls.append(result)
        raise RuntimeError(f"renderer rejected {secret}; NVIDIA_API_KEY={named_secret}")

    monkeypatch.setattr(workflow_module.widgets, "Output", _RecordingOutput)
    monkeypatch.setattr(workflow_module, "ipython_display", lambda value: value)
    workflow = workflow_module.launch_interactive_panel_design(
        FakeAgent(),
        expected_panel_size=24,
        on_complete=failing_callback,
    )

    workflow._approve_and_run(workflow.approve_button)

    assert workflow.status == "completed"
    assert workflow.agent_run is run
    assert workflow.result_output.enter_count == 1
    assert callback_calls == [run]
    assert workflow.result_output.rendered_errors == []
    assert secret not in workflow.transcript_text
    assert secret not in _widget_text(workflow.root)
    assert ".def/ghi+jkl=" not in workflow.transcript_text
    assert ".def/ghi+jkl=" not in _widget_text(workflow.root)
    assert named_secret not in workflow.transcript_text
    assert named_secret not in _widget_text(workflow.root)
    expected = "renderer rejected [REDACTED]; NVIDIA_API_KEY=[REDACTED]"
    assert expected in workflow.transcript_text
    assert expected in _widget_text(workflow.root)
    assert "Completion display failed" in workflow.transcript_text
    assert "Completion display failed" in _widget_text(workflow.root)


def test_module3_widget_redacts_every_failure_card_message(monkeypatch):
    agent = _load_agent()
    workflow_module = _load_workflow(agent)
    secret = "nvapi-abc.def/ghi+jkl="
    named_secret = "named.secret/value+tail="
    raw_message = f"provider exposed {secret} NVIDIA_API_KEY={named_secret}"

    monkeypatch.setattr(workflow_module.widgets, "Output", _RecordingOutput)
    workflow = workflow_module.InteractivePanelDesignWorkflow(
        object(), expected_panel_size=24
    )
    workflow._error_card("Plan request failed", RuntimeError(raw_message))
    workflow._progress(
        "attempt_failed",
        {"attempt": 1, "message": raw_message, "will_revise": False},
    )
    workflow._progress("audit_failed", {"message": raw_message})

    visible_text = _widget_text(workflow.root)
    for raw_secret in (secret, ".def/ghi+jkl=", named_secret):
        assert raw_secret not in visible_text
        assert raw_secret not in workflow.transcript_text
    expected = "provider exposed [REDACTED] NVIDIA_API_KEY=[REDACTED]"
    assert expected in visible_text
    assert expected in workflow.transcript_text


def test_module3_widget_labels_returned_failed_run_consistently(monkeypatch, tmp_path):
    agent = _load_agent()
    workflow_module = _load_workflow(agent)
    plan = agent.PanelPlan.model_validate(_panel_plan_payload())
    run = agent.PanelAgentRun(
        success=False,
        approved_strategy=2,
        attempts=(),
        analysis_path=tmp_path / "analysis.py",
        panel_path=tmp_path / "panel.csv",
        report_path=tmp_path / "report.json",
        trace_path=tmp_path / "agent_trace.json",
        audit=None,
    )

    class FakeAgent:
        def request_plan(self):
            return plan

        def run(self, **kwargs):
            return run

    callback_calls = []
    monkeypatch.setattr(workflow_module.widgets, "Output", _RecordingOutput)
    monkeypatch.setattr(workflow_module, "ipython_display", lambda value: value)
    workflow = workflow_module.launch_interactive_panel_design(
        FakeAgent(),
        expected_panel_size=24,
        on_complete=callback_calls.append,
    )

    workflow._approve_and_run(workflow.approve_button)

    visible_text = _widget_text(workflow.root)
    assert "<h3>Analysis did not validate</h3>" in visible_text
    assert "Analysis did not validate" in workflow.transcript_text
    assert "Agent workflow did not pass" not in visible_text
    assert "Agent workflow did not pass" not in workflow.transcript_text
    assert "Agent run stopped safely" not in visible_text
    assert "Agent run stopped safely" not in workflow.transcript_text
    assert "Analysis validated;" not in visible_text
    assert "Analysis validated;" not in workflow.transcript_text
    assert workflow.status == "failed"
    assert workflow.agent_run is run
    assert workflow.result_output.enter_count == 0
    assert callback_calls == []


def test_module3_widget_labels_raised_run_failure_consistently(monkeypatch):
    agent = _load_agent()
    workflow_module = _load_workflow(agent)
    plan = agent.PanelPlan.model_validate(_panel_plan_payload())
    secret = "nvapi-abc.def/ghi+jkl="
    named_secret = "named.secret/value+tail="

    class FakeAgent:
        def request_plan(self):
            return plan

        def run(self, **kwargs):
            raise RuntimeError(
                f"bounded run failed {secret}; NVIDIA_API_KEY={named_secret}"
            )

    callback_calls = []
    monkeypatch.setattr(workflow_module.widgets, "Output", _RecordingOutput)
    monkeypatch.setattr(workflow_module, "ipython_display", lambda value: value)
    workflow = workflow_module.launch_interactive_panel_design(
        FakeAgent(),
        expected_panel_size=24,
        on_complete=callback_calls.append,
    )

    workflow._approve_and_run(workflow.approve_button)

    visible_text = _widget_text(workflow.root)
    assert "<h3>Analysis did not validate</h3>" in visible_text
    assert "Analysis did not validate" in workflow.transcript_text
    assert "Agent workflow did not pass" not in visible_text
    assert "Agent workflow did not pass" not in workflow.transcript_text
    assert "Agent run stopped safely" not in visible_text
    assert "Agent run stopped safely" not in workflow.transcript_text
    assert "Analysis validated;" not in visible_text
    assert "Analysis validated;" not in workflow.transcript_text
    assert secret not in visible_text
    assert secret not in workflow.transcript_text
    assert ".def/ghi+jkl=" not in visible_text
    assert ".def/ghi+jkl=" not in workflow.transcript_text
    assert named_secret not in visible_text
    assert named_secret not in workflow.transcript_text
    expected = "bounded run failed [REDACTED]; NVIDIA_API_KEY=[REDACTED]"
    assert expected in visible_text
    assert expected in workflow.transcript_text
    assert workflow.status == "failed"
    assert workflow.agent_run is None
    assert workflow.result_output.enter_count == 0
    assert callback_calls == []


def test_module3_trace_records_reference_mode(monkeypatch, tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    monkeypatch.setattr(
        agent.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="bounded test failure"
        ),
    )

    run = panel_agent.run(approved_strategy=1, expected_panel_size=24)
    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))

    assert trace["mode"] == "reference"


@pytest.mark.parametrize("approved_strategy", [1, 2])
def test_both_panel_strategies_beat_the_fixed_first_24_baseline(
    tmp_path, approved_strategy
):
    agent = _load_agent()
    candidate_frame = _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()

    run = panel_agent.run(
        approved_strategy=approved_strategy,
        expected_panel_size=24,
        timeout_seconds=120,
    )

    assert run.success, run.attempts[0].message
    panel_rows = run.panel_path.read_text(encoding="utf-8").splitlines()
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))
    acceptance = report["acceptance"]
    candidate_keys = set(candidate_frame["canonical_ikey"].tolist())
    import pandas as pd

    selected_keys = set(pd.read_csv(run.panel_path)["canonical_ikey"].tolist())

    assert len(panel_rows) == 25
    assert report["candidate_count"] == 96
    assert report["panel_count"] == 24
    assert report["unique_ikeys"] == 24
    assert len(selected_keys) == 24
    assert selected_keys < candidate_keys
    assert (
        acceptance["selected_minimum_distance"]
        >= acceptance["baseline_minimum_distance"]
    )
    assert (
        acceptance["selected_descriptor_coverage"]
        >= acceptance["baseline_descriptor_coverage"]
    )
    assert (
        acceptance["selected_minimum_distance"]
        > acceptance["baseline_minimum_distance"]
        or acceptance["selected_descriptor_coverage"]
        > acceptance["baseline_descriptor_coverage"]
    )
    assert trace["mode"] == "reference"
    assert trace["model"] is None


def test_panel_validator_recomputes_all_candidate_descriptors(tmp_path):
    import pandas as pd

    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    run = panel_agent.run(
        approved_strategy=2,
        expected_panel_size=24,
        timeout_seconds=120,
    )
    assert run.success, run.attempts[0].message

    candidate_path = tmp_path / "reframe_candidates.csv"
    original = pd.read_csv(candidate_path)
    selected_keys = set(pd.read_csv(run.panel_path)["canonical_ikey"])
    accepted_forged_descriptors = []
    for column in ("MolWt", "cLogP", "TPSA"):
        forged = original.copy()
        eligible = forged.loc[
            ~forged["canonical_ikey"].isin(selected_keys)
            & forged[column].ne(forged[column].min())
            & forged[column].ne(forged[column].max())
        ]
        assert not eligible.empty
        forged_index = eligible.index[0]
        forged.loc[forged_index, column] += 0.123456
        forged.to_csv(candidate_path, index=False)
        try:
            agent.validate_panel_artifacts(tmp_path, expected_panel_size=24)
        except agent.WorkshopAgentError:
            pass
        else:
            accepted_forged_descriptors.append(column)

    assert accepted_forged_descriptors == []


def test_panel_validator_rejects_forged_values_in_every_report_category(tmp_path):
    agent = _load_agent()
    _write_candidate_workspace(tmp_path)
    panel_agent = agent.PanelDesignAgent(
        workdir=tmp_path,
        mission="bounded mission",
        mode="reference",
    )
    panel_agent.request_plan()
    run = panel_agent.run(
        approved_strategy=2,
        expected_panel_size=24,
        timeout_seconds=120,
    )
    assert run.success, run.attempts[0].message
    report_path = run.report_path
    original = json.loads(report_path.read_text(encoding="utf-8"))

    def alternate_backend(report):
        report["backend"] = (
            "nvmolkit-gpu"
            if report["backend"] != "nvmolkit-gpu"
            else "rdkit-cpu-reference (not GPU evidence)"
        )

    def forge_strategy(report):
        report["parameters"].update(
            {
                "strategy": "cluster_aware_max_min",
                "radius": 2,
                "fp_bits": 1024,
                "distance_cutoff": 0.55,
            }
        )

    def forge_raw_range(report):
        report["parameters"]["raw_similarity_range"] = [0.25, 0.75]

    def forge_distance_cutoff(report):
        report["parameters"]["distance_cutoff"] = 0.55

    def forge_quantiles(report):
        report["descriptor_quantiles"]["candidate"]["MolWt"]["median"] += 100.0

    def forge_pairwise(report):
        report["pairwise_similarity"].update(
            {"pair_count": 1, "median": 0.99, "p95": 0.99, "maximum": 0.99}
        )

    def forge_cluster_coverage(report):
        report["cluster_coverage"]["selected_compounds"] = 23

    def forge_limitations(report):
        report["limitations"] = ["Forged limitation"]

    forgeries = {
        "backend": alternate_backend,
        "strategy": forge_strategy,
        "raw_similarity_range": forge_raw_range,
        "distance_cutoff": forge_distance_cutoff,
        "descriptor_quantiles": forge_quantiles,
        "pairwise_similarity": forge_pairwise,
        "cluster_coverage": forge_cluster_coverage,
        "limitations": forge_limitations,
    }
    accepted_forgeries = []
    for label, mutate in forgeries.items():
        forged = json.loads(json.dumps(original))
        mutate(forged)
        report_path.write_text(json.dumps(forged), encoding="utf-8")
        try:
            agent.validate_panel_artifacts(tmp_path, expected_panel_size=24)
        except agent.WorkshopAgentError:
            pass
        else:
            accepted_forgeries.append(label)

    assert accepted_forgeries == []


def test_panel_cluster_membership_is_key_bound_but_label_invariant():
    agent = _load_agent()
    independent = {"key-a": 0, "key-b": 0, "key-c": 1, "key-d": 1}
    globally_renumbered = [
        {"canonical_ikey": "key-a", "method_cluster": "900"},
        {"canonical_ikey": "key-b", "method_cluster": "900"},
        {"canonical_ikey": "key-c", "method_cluster": "100"},
        {"canonical_ikey": "key-d", "method_cluster": "100"},
    ]
    swapped_across_clusters = [
        {"canonical_ikey": "key-a", "method_cluster": "900"},
        {"canonical_ikey": "key-b", "method_cluster": "100"},
        {"canonical_ikey": "key-c", "method_cluster": "900"},
        {"canonical_ikey": "key-d", "method_cluster": "100"},
    ]

    assert agent._panel_cluster_membership_matches(globally_renumbered, independent)
    assert not agent._panel_cluster_membership_matches(
        swapped_across_clusters, independent
    )
