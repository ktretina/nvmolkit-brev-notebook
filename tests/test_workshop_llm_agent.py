import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import ast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "notebooks" / "workshop_llm_agent.py"
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "02_agent_assisted_reframe_neighborhoods.ipynb"
)


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
    assert call["model"] == "nvidia/nemotron-3-nano-30b-a3b"
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
        "## Step 3 — Validate and bind the locally rendered function",
        "## Step 4 — Run the bound function and its acceptance tests",
        "## Step 6 — Review the local receipt and results",
    ):
        assert required_text in source


def test_module2_notebook_has_one_recoverable_bounded_attendee_flow():
    source = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    for required_text in (
        "hosted nemotron returns only two bounded policy choices plus explanations",
        "python renders, validates, and binds the function",
        "set `nvmolkit_workshop_mode=reference`, restart, and rerun the notebook",
        "zero client calls",
        "## step 4 — run the bound function and its acceptance tests",
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
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    discussion = source[
        source.index("## Checks and next step") : source.index(
            "## Sources and scientific boundary"
        )
    ]
    answer_key = source[source.index("## Answer key") :]
    assert "selected failure policy" in discussion.lower()
    assert "radius-sensitive anchor" in discussion.lower()
    assert "unsupported scientific inference" in discussion.lower()
    assert "selected `skip` would continue" in answer_key
    assert "lowest top-10 Jaccard overlap" in answer_key
    assert "binding, activity, ADMET, efficacy, or safety" in answer_key
