import http.server
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_KEY_SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
RENDER_SETUP_PATH = REPO_ROOT / "launchable" / "render_setup.py"


def _load_render_setup():
    assert RENDER_SETUP_PATH.is_file(), "launchable/render_setup.py is required"
    spec = importlib.util.spec_from_file_location(
        "nvmolkit_render_setup", RENDER_SETUP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_preserves_launch_and_separate_acceptance_gates():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    renderer_command = (
        "python3 launchable/render_setup.py "
        "/private/tmp/nvmolkit-workshop-setup.sh"
    )
    for instructions in (readme, fields):
        assert "Brev-managed Jupyter" in instructions
        assert "Only my organization" in instructions
        assert "Secure Link" in instructions
        assert "CPython 3.12" in instructions
        assert "mode `0600`" in instructions
        assert "No Launch parameters or Setup values" in instructions
        assert "required Text parameter" not in instructions
        assert "no default" not in instructions.lower()
        assert "paste the current contents of `launchable/setup.sh`" not in instructions
        assert "short-lived setup environment" not in instructions
        assert "legacy variable name" not in instructions
        assert "entered once in Brev Setup values" not in instructions
        assert "Enter the supplied value once in Setup values" not in instructions
        assert renderer_command in instructions
        assert "hidden prompt" in instructions.lower()
        assert "outside the repository" in instructions
        assert "paste only the rendered file" in instructions.lower()
        assert "delete the private rendered file after saving" in instructions.lower()
        assert "controls a deployed VM can recover" in instructions
        assert "rotate or revoke" in instructions
    lowered = readme.lower()
    for gate in ("local deterministic acceptance", "gpu acceptance", "hosted inference acceptance", "rendered deployment acceptance"):
        assert gate in lowered
    assert "bounded policy" in lowered
    assert "strict plan" in lowered
    assert "strict audit" in lowered
    assert "minimum tanimoto distance" in lowered
    assert "aggregate input profile" in lowered
    assert "independently validated aggregate report snapshot" in lowered
    assert "no raw molecule rows" in lowered
    assert "credentials" in lowered and "local visualization artifacts" in lowered
    assert "pytest -q" in readme
    assert "RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q" in readme
    assert "not yet live-qualified" in lowered
    assert "Attendees enter no API key" in readme
    assert "do not need an NVIDIA API account or key" in readme


def test_readme_persistence_receipt_requires_no_credential_reentry():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    persistence_receipt = next(
        line
        for line in readme.splitlines()
        if line.startswith("- **Persistence receipt:**")
    )
    lowered = persistence_receipt.lower()

    assert "protected credential persistence" in lowered
    assert "no credential re-entry" in lowered
    assert "credential-reentry" not in lowered
    assert "credential re-entry checks" not in lowered


def test_plan_routes_fake_key_rendering_through_the_trusted_renderer():
    plan = (
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-08-21-zero-input-workshop-key.md"
    ).read_text(encoding="utf-8")

    assert "setup_source.replace(SETUP_KEY_SENTINEL, rendered_key)" not in plan
    assert "renderer.render_setup(setup_source, rendered_key)" in plan
    assert "`launchable/render_setup.py`" in plan
    assert re.search(r"pure\s+renderer validates", plan)
    assert "`shlex.quote`" in plan
    assert "fake keys" in plan
    assert (
        "result, fake_home, log = _run_setup(tmp_path, rendered_key=invalid_key)"
        not in plan
    )
    assert "test_renderer_rejects_invalid_prefix_before_script_exists" in plan
    assert 'with pytest.raises(ValueError, match="beginning with sk-"):' in plan
    assert 'tmp_path / "brev-generated-setup.sh"' in plan
    assert 'tmp_path / "invocations.log"' in plan


def test_setup_uses_brev_managed_python_and_leaves_jupyter_to_brev():
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert setup.splitlines()[0] == "#!/bin/bash"
    assert setup.splitlines()[1:3] == ["set +x +v", "set -euo pipefail"]
    assert setup.count(SETUP_KEY_SENTINEL) == 1
    assert f"launch_api_key={SETUP_KEY_SENTINEL}" in setup
    assert "NVMOLKIT_EMBEDDED_CREDENTIAL_EOF" not in setup
    assert "required in Brev Setup values" not in setup
    assert not re.search(
        r'launch_api_key=.*\$\{NVIDIA_(?:INFERENCE_)?API_KEY', setup
    )
    assert '${HOME}/.venv/bin/python3' in setup
    assert "command -v python3.12" not in setup
    assert '"${PYTHON}" -m pip --version' in setup
    assert '"${PYTHON}" -m ensurepip --upgrade' in setup
    assert '"${PYTHON}" -m pip install --upgrade pip' in setup
    assert '"${PYTHON}" -m pip install -r requirements.txt' in setup
    assert 'url = "http://127.0.0.1:8888/api"' in setup
    assert '${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY' in setup
    assert '${HOME}/.jupyter/lab/user-settings/@jupyter-widgets/jupyterlab-manager' in setup
    assert '"saveState": true' in setup
    assert 'chmod 600 "${api_key_temp}"' in setup
    assert 'printf \'%s\' "${launch_api_key}" >"${api_key_temp}"' in setup
    assert 'mv -f -- "${api_key_temp}" "${api_key_path}"' in setup
    assert "NEMOTRON_MODEL" not in setup
    assert "JUPYTER_PORT" not in setup
    assert len(setup.encode("utf-8")) <= 16_384
    source_cleanup = setup.index("unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY")
    assert source_cleanup < setup.index('if [[ -f "${PWD}/requirements.txt"')
    assert source_cleanup < setup.index('install -d -m 700 "${api_key_directory}"')
    key_persistence = setup.index(
        'mv -f -- "${api_key_temp}" "${api_key_path}"'
    )
    widget_settings = setup.index('widget_settings_directory="${HOME}/.jupyter')
    assert key_persistence < setup.index("unset launch_api_key", key_persistence)
    assert setup.index("unset launch_api_key", key_persistence) < widget_settings
    assert not re.search(
        r"(?m)^\s*export\s+(?:launch_api_key|NVIDIA_(?:INFERENCE_)?API_KEY)",
        setup,
    )
    assert all(forbidden not in setup for forbidden in ("jupyter lab", "nohup", "PID_FILE", "kill ", "-m venv"))
    assert "jupyterlab==" not in requirements


def test_renderer_shell_quotes_accepted_single_line_key():
    renderer = _load_render_setup()
    key = "sk-quote-test'; $(printf unsafe) # spaces"
    template = f"launch_api_key={SETUP_KEY_SENTINEL}\n"

    rendered = renderer.render_setup(template, key)

    assert rendered == f"launch_api_key={shlex.quote(key)}\n"
    assert SETUP_KEY_SENTINEL not in rendered


@pytest.mark.parametrize(
    "template",
    [
        "launch_api_key=missing\n",
        f"{SETUP_KEY_SENTINEL}\n{SETUP_KEY_SENTINEL}\n",
    ],
)
def test_renderer_requires_exactly_one_sentinel(template):
    renderer = _load_render_setup()

    with pytest.raises(ValueError, match="exactly one"):
        renderer.render_setup(template, "sk-test-key")


@pytest.mark.parametrize(
    "key",
    [
        "nvapi-invalid-prefix",
        "sk-",
        "sk-carriage\rreturn",
        "sk-line\nfeed",
        "sk-nul\x00byte",
    ],
)
def test_renderer_rejects_invalid_or_multiline_keys(key):
    renderer = _load_render_setup()
    template = f"launch_api_key={SETUP_KEY_SENTINEL}\n"

    with pytest.raises(ValueError):
        renderer.render_setup(template, key)


def test_renderer_cli_writes_private_file_without_key_output(
    monkeypatch,
    tmp_path,
    capsys,
):
    renderer = _load_render_setup()
    key = "sk-cli-private-sentinel-must-not-leak"
    output_path = tmp_path / "private-rendered-setup.sh"
    monkeypatch.setattr(renderer.getpass, "getpass", lambda _prompt: key)

    assert renderer.main([str(output_path)]) == 0

    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert str(output_path.resolve()) in captured.out
    assert output_path.stat().st_mode & 0o777 == 0o600
    template = (REPO_ROOT / "launchable" / "setup.sh").read_text(
        encoding="utf-8"
    )
    assert output_path.read_text(encoding="utf-8") == renderer.render_setup(
        template,
        key,
    )


def test_renderer_cli_refuses_existing_output_before_prompt(
    monkeypatch,
    tmp_path,
):
    renderer = _load_render_setup()
    output_path = tmp_path / "existing-rendered-setup.sh"
    output_path.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(
        renderer.getpass,
        "getpass",
        lambda _prompt: pytest.fail("existing output must be rejected before prompt"),
    )

    with pytest.raises(SystemExit) as exc_info:
        renderer.main([str(output_path)])

    assert exc_info.value.code == 2
    assert output_path.read_text(encoding="utf-8") == "preserve me"


def test_renderer_cli_refuses_output_inside_repository_before_prompt(
    monkeypatch,
):
    renderer = _load_render_setup()
    output_path = REPO_ROOT / "forbidden-rendered-setup-test.sh"
    assert not output_path.exists()
    monkeypatch.setattr(
        renderer.getpass,
        "getpass",
        lambda _prompt: pytest.fail("repository output must be rejected before prompt"),
    )

    with pytest.raises(SystemExit) as exc_info:
        renderer.main([str(output_path)])

    assert exc_info.value.code == 2
    assert not output_path.exists()


def test_renderer_cli_refuses_dangling_output_symlink(monkeypatch, tmp_path):
    renderer = _load_render_setup()
    output_path = tmp_path / "dangling-rendered-setup.sh"
    symlink_target = tmp_path / "missing-symlink-target.sh"
    output_path.symlink_to(symlink_target)
    prompt_count = 0

    def read_key(_prompt):
        nonlocal prompt_count
        prompt_count += 1
        return "sk-dangling-symlink-test"

    monkeypatch.setattr(renderer.getpass, "getpass", read_key)
    exit_code = None
    try:
        renderer.main([str(output_path)])
    except SystemExit as error:
        exit_code = error.code

    assert not symlink_target.exists()
    assert exit_code == 2
    assert prompt_count == 0
    assert os.path.lexists(output_path)


def test_renderer_cli_refuses_symlink_parent(monkeypatch, tmp_path):
    renderer = _load_render_setup()
    real_parent = tmp_path / "real-parent"
    symlink_parent = tmp_path / "symlink-parent"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    output_path = symlink_parent / "private-rendered-setup.sh"
    prompt_count = 0

    def read_key(_prompt):
        nonlocal prompt_count
        prompt_count += 1
        return "sk-symlink-parent-test"

    monkeypatch.setattr(renderer.getpass, "getpass", read_key)
    exit_code = None
    try:
        renderer.main([str(output_path)])
    except SystemExit as error:
        exit_code = error.code

    assert not (real_parent / output_path.name).exists()
    assert exit_code == 2
    assert prompt_count == 0


@pytest.mark.parametrize("parent_kind", ["missing", "file"])
def test_renderer_cli_requires_existing_directory_parent_before_prompt(
    monkeypatch,
    tmp_path,
    parent_kind,
):
    renderer = _load_render_setup()
    output_parent = tmp_path / f"{parent_kind}-parent"
    if parent_kind == "file":
        output_parent.write_text("not a directory", encoding="utf-8")
    output_path = output_parent / "private-rendered-setup.sh"
    prompt_count = 0

    def read_key(_prompt):
        nonlocal prompt_count
        prompt_count += 1
        return "sk-invalid-parent-test"

    monkeypatch.setattr(renderer.getpass, "getpass", read_key)

    with pytest.raises(SystemExit) as exc_info:
        renderer.main([str(output_path)])

    assert exc_info.value.code == 2
    assert prompt_count == 0
    assert not os.path.lexists(output_path)


def test_renderer_cli_refuses_repository_symlink_alias_before_prompt(
    monkeypatch,
    tmp_path,
):
    renderer = _load_render_setup()
    repository_alias = tmp_path / "repository-alias"
    repository_alias.symlink_to(REPO_ROOT, target_is_directory=True)
    output_path = (
        repository_alias / "launchable" / "forbidden-alias-rendered-setup.sh"
    )
    assert not os.path.lexists(output_path)
    monkeypatch.setattr(
        renderer.getpass,
        "getpass",
        lambda _prompt: pytest.fail("repository alias must fail before prompt"),
    )

    with pytest.raises(SystemExit) as exc_info:
        renderer.main([str(output_path)])

    assert exc_info.value.code == 2
    assert not os.path.lexists(output_path)


def test_renderer_cli_refuses_nested_repository_symlink_alias_before_prompt(
    monkeypatch,
    tmp_path,
):
    renderer = _load_render_setup()
    nested_repository_parent = REPO_ROOT / "launchable" / f".{tmp_path.name}"
    nested_repository_parent.mkdir()
    repository_subdirectory_alias = tmp_path / "launchable-alias"
    repository_subdirectory_alias.symlink_to(
        REPO_ROOT / "launchable",
        target_is_directory=True,
    )
    output_path = (
        repository_subdirectory_alias
        / nested_repository_parent.name
        / "forbidden-nested-alias-rendered-setup.sh"
    )
    physical_output_path = nested_repository_parent / output_path.name
    try:
        assert not os.path.lexists(output_path)
        monkeypatch.setattr(
            renderer.getpass,
            "getpass",
            lambda _prompt: pytest.fail(
                "nested repository alias must fail before prompt"
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            renderer.main([str(output_path)])

        assert exc_info.value.code == 2
        assert not os.path.lexists(output_path)
    finally:
        if os.path.lexists(physical_output_path):
            physical_output_path.unlink()
        nested_repository_parent.rmdir()


def test_renderer_cli_refuses_case_alias_of_repository_when_supported(
    monkeypatch,
):
    renderer = _load_render_setup()
    repository_text = os.fspath(REPO_ROOT)
    repository_alias = None
    for index, character in enumerate(repository_text):
        alternate = character.swapcase()
        if alternate == character:
            continue
        candidate = Path(
            repository_text[:index] + alternate + repository_text[index + 1 :]
        )
        try:
            is_alias = os.path.samefile(candidate, REPO_ROOT)
        except OSError:
            continue
        if candidate != REPO_ROOT and is_alias:
            repository_alias = candidate
            break
    if repository_alias is None:
        pytest.skip("filesystem does not expose an alternate-case repository alias")

    output_path = repository_alias / "forbidden-case-rendered-setup.sh"
    assert not os.path.lexists(output_path)
    monkeypatch.setattr(
        renderer.getpass,
        "getpass",
        lambda _prompt: pytest.fail("case alias must fail before prompt"),
    )

    with pytest.raises(SystemExit) as exc_info:
        renderer.main([str(output_path)])

    assert exc_info.value.code == 2
    assert not os.path.lexists(output_path)


def test_renderer_cli_rejects_parent_swap_without_redirecting_output(
    monkeypatch,
    tmp_path,
):
    renderer = _load_render_setup()
    output_parent = tmp_path / "verified-parent"
    displaced_parent = tmp_path / "displaced-parent"
    replacement_parent = tmp_path / "replacement-parent"
    output_parent.mkdir()
    replacement_parent.mkdir()
    (replacement_parent / "replacement-marker").write_text(
        "replacement",
        encoding="utf-8",
    )
    output_path = output_parent / "private-rendered-setup.sh"
    real_open = renderer.os.open
    swapped = False

    def swap_parent_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped:
            swapped = True
            output_parent.rename(displaced_parent)
            replacement_parent.rename(output_parent)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(renderer.os, "open", swap_parent_then_open)
    monkeypatch.setattr(
        renderer.getpass,
        "getpass",
        lambda _prompt: "sk-parent-swap-test",
    )
    exit_code = None
    try:
        renderer.main([str(output_path)])
    except SystemExit as error:
        exit_code = error.code

    assert swapped
    assert (output_parent / "replacement-marker").exists()
    assert not (output_parent / output_path.name).exists()
    assert not (displaced_parent / output_path.name).exists()
    assert exit_code == 2


def _run_setup(tmp_path, rendered_key=None, setup_values=None, bash_flags=None):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / "nvmolkit-brev-notebook").symlink_to(REPO_ROOT, target_is_directory=True)
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
  -m) printf 'MODULE %s %s %s\n' "${2:-}" "${3:-}" "${4:-}" >>"${INVOCATION_LOG}"; [[ "${2:-} ${3:-}" == "pip --version" ]] && exit 1; [[ "${4:-}" == "-r" && ! -f "${5:-}" ]] && exit 93; true ;;
  -) payload="$(</dev/stdin)"; [[ "$payload" == *"torch.cuda.is_available"* ]] && printf 'SMOKE\n' >>"${INVOCATION_LOG}" || printf 'HEALTH\n' >>"${INVOCATION_LOG}" ;;
  *) exit 92 ;;
esac
""", encoding="utf-8")
    managed_python.chmod(0o755)
    (fake_bin / "uname").write_text("""#!/bin/bash
set -euo pipefail
[[ -z "${NVIDIA_INFERENCE_API_KEY+x}" ]]
[[ -z "${NVIDIA_API_KEY+x}" ]]
[[ -z "${launch_api_key+x}" ]]
printf 'ENV_CLEAN\n' >>"${INVOCATION_LOG}"
[[ "${1:-}" == "-s" ]] && printf 'Linux\n' || printf 'x86_64\n'
""", encoding="utf-8")
    (fake_bin / "uname").chmod(0o755)
    base_env = {
        name: value
        for name, value in os.environ.items()
        if name not in {"NVIDIA_INFERENCE_API_KEY", "NVIDIA_API_KEY"}
    }
    env = base_env | {
        "HOME": str(fake_home),
        "INVOCATION_LOG": str(log),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
    } | (setup_values or {})
    copied_setup = tmp_path / "brev-generated-setup.sh"
    setup = (REPO_ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    if rendered_key is not None:
        setup = _load_render_setup().render_setup(setup, rendered_key)
    copied_setup.write_text(setup, encoding="utf-8")
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    result = subprocess.run(
        ["bash", *(bash_flags or ()), str(copied_setup)],
        cwd=execution_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, fake_home, log


def test_unrendered_setup_fails_before_installation(tmp_path):
    result, fake_home, log = _run_setup(tmp_path)

    assert result.returncode != 0
    assert "private Brev Console copy" in result.stderr
    assert SETUP_KEY_SENTINEL not in result.stdout + result.stderr
    assert not log.exists()
    assert not (
        fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_rendered_setup_ignores_ambient_keys_and_runs_managed_runtime(tmp_path):
    rendered_key = "sk-rendered-setup-sentinel-must-not-leak"
    ambient_primary = "sk-ambient-primary-sentinel-must-not-leak"
    ambient_legacy = "sk-ambient-legacy-sentinel-must-not-leak"
    result, fake_home, log = _run_setup(
        tmp_path,
        rendered_key=rendered_key,
        setup_values={
            "NVIDIA_INFERENCE_API_KEY": ambient_primary,
            "NVIDIA_API_KEY": ambient_legacy,
        },
    )
    assert result.returncode == 0, result.stderr
    key_directory = fake_home / ".config" / "nvmolkit"
    key_file = key_directory / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == rendered_key
    assert key_directory.stat().st_mode & 0o777 == 0o700
    assert key_file.stat().st_mode & 0o777 == 0o600
    widget_settings = (
        fake_home
        / ".jupyter"
        / "lab"
        / "user-settings"
        / "@jupyter-widgets"
        / "jupyterlab-manager"
        / "plugin.jupyterlab-settings"
    )
    assert json.loads(widget_settings.read_text(encoding="utf-8")) == {
        "saveState": True
    }
    combined_output = result.stdout + result.stderr
    for fake_secret in (rendered_key, ambient_primary, ambient_legacy):
        assert fake_secret not in combined_output
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert invocations.count("ENV_CLEAN") == 2
    assert any("sys.implementation.name" in line for line in invocations)
    assert invocations.index("MODULE ensurepip --upgrade ") < invocations.index("MODULE pip install --upgrade")
    assert invocations.index("MODULE pip install --upgrade") < invocations.index("MODULE pip install -r")
    assert invocations.index("MODULE pip install -r") < invocations.index("SMOKE") < invocations.index("HEALTH")


def test_rendered_setup_treats_shell_syntax_as_literal_key_data(tmp_path):
    marker = tmp_path / "credential-injection-marker"
    rendered_key = (
        "sk-injection-test'; "
        f'printf injected >"{marker}"; '
        "launch_api_key='sk-after-injection"
    )

    result, fake_home, _ = _run_setup(tmp_path, rendered_key=rendered_key)

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    key_file = fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == rendered_key
    assert rendered_key not in result.stdout + result.stderr


def test_rendered_setup_disables_trace_and_verbose_before_reading_key(tmp_path):
    rendered_key = "sk-trace-verbose-sentinel-must-not-leak"

    result, fake_home, _ = _run_setup(
        tmp_path,
        rendered_key=rendered_key,
        bash_flags=("-xv",),
    )

    assert result.returncode == 0, result.stderr
    key_file = fake_home / ".config" / "nvmolkit" / "NVIDIA_INFERENCE_API_KEY"
    assert key_file.read_text(encoding="utf-8") == rendered_key
    assert rendered_key not in result.stdout + result.stderr


def test_renderer_rejects_multiline_delimiter_breakout_before_script_exists(
    tmp_path,
):
    marker = tmp_path / "multiline-injection-marker"
    rendered_key = (
        "sk-multiline-test\n"
        "NVMOLKIT_EMBEDDED_CREDENTIAL_EOF\n"
        f'printf injected >"{marker}"\n'
        "exit 0"
    )

    with pytest.raises(ValueError, match="one line"):
        _run_setup(tmp_path, rendered_key=rendered_key)

    assert not marker.exists()
    assert not (tmp_path / "brev-generated-setup.sh").exists()


def test_renderer_rejects_invalid_prefix_before_script_exists(tmp_path):
    invalid_key = "nvapi-rendered-sentinel-must-not-leak"

    with pytest.raises(ValueError, match="beginning with sk-"):
        _run_setup(tmp_path, rendered_key=invalid_key)

    assert not (tmp_path / "brev-generated-setup.sh").exists()
    assert not (tmp_path / "invocations.log").exists()
    assert not (
        tmp_path
        / "home"
        / ".config"
        / "nvmolkit"
        / "NVIDIA_INFERENCE_API_KEY"
    ).exists()


def test_launchable_contract_fixes_storage_model_port_and_zero_setup_values():
    fields = (REPO_ROOT / "launchable" / "fields.md").read_text(encoding="utf-8")
    assert "75 GiB" in fields
    assert "50 GiB" not in fields
    assert "No Launch parameters or Setup values" in fields
    assert "required Text parameter" not in fields
    assert "no default" not in fields.lower()
    assert "`nvidia/nvidia/nemotron-3-nano-30b-a3b`" in fields
    assert "`https://inference-api.nvidia.com/v1`" in fields
    assert "port `8888`" in fields
    assert (
        "Remove `NVIDIA_INFERENCE_API_KEY`, `NVIDIA_API_KEY`, "
        "`NEMOTRON_MODEL`, and `JUPYTER_PORT`" in fields
    )
    assert "redacted operator template" in fields
    assert (
        "python3 launchable/render_setup.py "
        "/private/tmp/nvmolkit-workshop-setup.sh" in fields
    )
    assert "hidden prompt" in fields.lower()
    assert "outside the repository" in fields
    assert "mode `0600`" in fields
    assert "paste only the rendered file" in fields.lower()
    assert "delete the private rendered file after saving" in fields.lower()
    assert "controls a deployed VM can recover" in fields
    assert "rotate or revoke" in fields
    assert "paste the current contents of `launchable/setup.sh`" not in fields


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
