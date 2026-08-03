# Hosted Fingerprint Arguments Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first Nemotron-selected nvMolKit operation execute reliably without changing the notebook narrative, scientific workflow, or fail-closed tool contract.

**Architecture:** Keep the existing eight-cell notebook and one persistent Nemotron conversation. First expose only secret-safe Pydantic error locations and types for the failing `generate_morgan_fingerprints` call; then make exactly one schema correction supported by that live evidence. Do not add retries, fallback parameters, aliases, an agent framework, or notebook code.

**Tech Stack:** Python 3.12, OpenAI-compatible NVIDIA hosted inference, Pydantic v2, pytest, Brev L4, nvMolKit 0.5.0.

---

## Diagnosis already established

The attached run proves:

- hosted authentication works because Nemotron returns the six-stage plan;
- RDKit input inspection works and reports 256 valid rows plus a 24-molecule preview;
- the failure occurs on the next eligible stage, `generate_morgan_fingerprints`;
- `MorganFingerprintGenerator` never executes because `FingerprintArgs.model_validate(...)` raises first;
- the current exception deliberately hides the Pydantic field/type details.

The leading hypotheses are a returned key vocabulary that differs from the current `radius` / `size` schema, or a `decision_basis` string that violates the current 12–240-character single-line pattern. The PDF cannot distinguish those cases, so changing either now would be guesswork.

### Task 1: Expose a secret-safe validation signature

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`

- [ ] **Step 1: Add failing tests for safe validation details**

Add one parameterized test using fake fingerprint calls that fail for:

```python
(
    {"fingerprint_radius": 2, "fingerprint_size": 1024,
     "decision_basis": "Use a compact molecular representation."},
    {"radius", "size", "fingerprint_radius", "fingerprint_size"},
)
(
    {"radius": 2, "size": 1024, "decision_basis": "short"},
    {"decision_basis"},
)
```

Require the exception to contain only the stage name, failing field locations, and Pydantic error types. Assert that it contains neither argument values nor `NVIDIA_API_KEY`, and that the fingerprint executor was not called.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python -m pytest tests/test_demo_agent.py -k "validation_signature or fingerprint" -v
```

Expected: the existing generic `_HostedArgumentsValidationError` does not expose the safe signature.

- [ ] **Step 3: Preserve Pydantic locations/types without preserving inputs**

Give the private exception a safe, value-free representation:

```python
class _HostedArgumentsValidationError(ToolCallError):
    def __init__(self, stage: str, issues: tuple[tuple[str, str], ...]):
        self.stage = stage
        self.issues = issues
        signature = ", ".join(f"{field}:{error_type}" for field, error_type in issues)
        super().__init__(f"{stage} arguments failed validation: {signature}")
```

Then change the `except ValidationError` branch to derive only safe metadata:

```python
except ValidationError as error:
    issues = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "type": item["type"],
        }
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    ]
    raise _HostedArgumentsValidationError(
        expected_name,
        tuple((item["field"], item["type"]) for item in issues),
    ) from None
```

Format its displayed message as a concise signature such as:

```text
generate_morgan_fingerprints arguments failed validation: decision_basis:string_too_short
```

Do not log raw arguments, hosted response bodies, prompts, or credentials.

- [ ] **Step 4: Run focused and full local tests**

Run:

```bash
python -m pytest tests/test_demo_agent.py -v
python -m pytest -q
python -m ruff check .
```

Expected: all deterministic tests pass and only the explicit GPU test skips locally.

- [ ] **Step 5: Commit the diagnostic only**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "fix: report safe hosted argument validation details"
```

### Task 2: Capture one live failure signature and make one correction

**Files:**
- Modify: `demo_agent.py`
- Modify: `tests/test_demo_agent.py`

- [ ] **Step 1: Deploy Task 1 to the existing Brev checkout and rerun once**

Use the existing guarded fast-forward procedure. In JupyterLab, close the stale tab without saving, reopen the notebook, restart the kernel, and run all cells with the hidden Developer API key.

Record only the safe signature. Stop after this run; do not stack speculative fixes.

- [ ] **Step 2: Apply exactly the matching correction**

Use this decision table:

| Safe signature | Minimal correction |
| --- | --- |
| missing `radius` / `size` plus extra `fingerprint_radius` / `fingerprint_size` | Rename `FingerprintArgs` fields to `fingerprint_radius` and `fingerprint_size`; update `_executor_arguments`, rendered-argument tests, and fake hosted calls. Use one canonical vocabulary; do not accept aliases. |
| missing `radius` / `size` plus extra `fpSize` | Keep Python executor translation to nvMolKit's `fpSize`, but expose the canonical hosted fields `fingerprint_radius` and `fingerprint_size`. Do not expose nvMolKit constructor spelling as agent JSON. |
| `decision_basis:string_too_short`, `string_too_long`, or `string_pattern_mismatch` | Replace the presentation-only constraint with stripped, non-empty text capped at 320 characters. Keep the prompt asking for one concise sentence. Do not change scientific parameter validation. |
| strict integer/type error on radius or size | Keep integer strictness. Strengthen the tool field descriptions to say “JSON integer” and rerun once; do not coerce strings into scientific parameters. |
| any other signature | Stop and inspect that exact field/type before changing code. |

For the likely canonical-field correction, the public model should be:

```python
class FingerprintArgs(_StrictModel):
    fingerprint_radius: Literal[2, 3]
    fingerprint_size: Literal[1024, 2048]
    decision_basis: DecisionBasis
```

and the executor adapter should remain explicit:

```python
return {
    "fingerprint_radius": arguments.fingerprint_radius,
    "fingerprint_size": arguments.fingerprint_size,
}
```

- [ ] **Step 3: Prove RED then GREEN for the captured signature**

Add the exact safe live signature as a fake hosted response. Confirm it fails before the correction, apply the one correction, then confirm it validates and calls the fingerprint executor once with the bounded numeric values.

- [ ] **Step 4: Run the local regression suite**

```bash
python -m pytest tests/test_demo_agent.py tests/test_notebook.py -v
python -m pytest -q
python -m ruff check .
```

Expected: all deterministic tests pass; the notebook remains eight cells with one `run_workflow(...)` call and no new visible implementation code.

- [ ] **Step 5: Commit only the evidence-supported fix**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "fix: align hosted fingerprint tool arguments"
```

### Task 3: Qualify the continuous hosted workflow

**Files:**
- Modify only if this acceptance run identifies a task-owned defect.

- [ ] **Step 1: Review the final diff boundary**

Require changes only in `demo_agent.py` and `tests/test_demo_agent.py`. The notebook, chemistry workflow, data, setup, vendored skill, dependencies, and GPU acceptance test should remain unchanged.

- [ ] **Step 2: Push and fast-forward the existing Brev checkout**

Push standalone `main`, verify the public and remote commit hashes match, and require a clean Brev checkout. Do not create, stop, reset, or delete an instance.

- [ ] **Step 3: Run the deterministic remote tests**

```bash
cd /home/ubuntu/nvmolkit-brev-notebook
/home/ubuntu/.venv/bin/python -m pytest -q
```

Expected: all deterministic tests pass with only the explicit live-GPU test skipped.

- [ ] **Step 4: Run the notebook from a fresh kernel**

Acceptance requires all of the following in one run:

1. actual Nemotron plan appears;
2. RDKit inspection and the 24-molecule preview appear;
3. `Nemotron → nvMolKit MorganFingerprintGenerator` appears with bounded parameters and a concise decision basis;
4. `crossTanimotoSimilarity`, `fused_butina`, `EmbedMolecules`, and `MMFFOptimizeMoleculesConfs` execute in order;
5. each completed result and figure appears immediately;
6. the evidence-linked, schema-checked synthesis cites E01–E06;
7. no automatic retry, deterministic parameter fallback, credential output, or raw hosted payload is shown.

- [ ] **Step 5: Stop at the correct proof boundary**

The hosted/rendered run proves this notebook execution with the supplied key and current model endpoint. It does not prove unrestricted autonomy, qualitative scientific truth, performance, binding, activity, ADMET, safety, or experimental conformations.

## Explicit non-goals

- no notebook cell changes;
- no agent framework, MCP server, or retry loop;
- no default parameter fallback;
- no permissive coercion of strings to scientific numbers;
- no duplicate argument aliases unless live evidence proves the endpoint cannot emit one canonical schema;
- no changes to nvMolKit execution, figures, evidence E01–E06, setup, data, or Launchable configuration.
