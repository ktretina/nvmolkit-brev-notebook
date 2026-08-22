from __future__ import annotations

import os
import importlib
import shlex
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "launchable" / "acs_console_bootstrap.sh.in"
PUBLIC_BOOTSTRAP = ROOT / "launchable" / "acs_console_bootstrap.sh"
AUTHORING_SHEET = ROOT / "launchable" / "ACS_LAUNCHABLE_FIELDS.md"
PLACEHOLDER = "@REVIEWED_PUBLIC_COMMIT_SHA@"
SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
REPO_URL = "https://github.com/ktretina/nvmolkit-brev-notebook.git"
DUMMY_COMMIT = "a" * 40
PINNED_SETUP_COMMIT = "ccd3d80093a7c161c4572a04e5661429c7eb8b87"


def _rendered_bootstrap(key: str | None = None) -> str:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert source.count(PLACEHOLDER) == 1
    rendered = source.replace(PLACEHOLDER, DUMMY_COMMIT)
    if key is not None:
        assert rendered.count(SENTINEL) == 1
        rendered = rendered.replace(SENTINEL, shlex.quote(key), 1)
    return rendered


def test_public_console_bootstrap_is_exact_pinned_and_below_brev_limit() -> None:
    expected = TEMPLATE.read_text(encoding="utf-8").replace(
        PLACEHOLDER, PINNED_SETUP_COMMIT
    )
    published = PUBLIC_BOOTSTRAP.read_text(encoding="utf-8")

    assert published == expected
    assert published.startswith("#!/usr/bin/env bash\n")
    assert published.count(PINNED_SETUP_COMMIT) == 1
    assert PLACEHOLDER not in published
    assert published.count(SENTINEL) == 1
    assert len(published.encode("utf-8")) <= 16_384

    completed = subprocess.run(
        ["bash", "-n", str(PUBLIC_BOOTSTRAP)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_console_bootstrap_is_syntax_valid_and_pins_a_detached_checkout(
    tmp_path: Path,
) -> None:
    rendered = _rendered_bootstrap()
    script = tmp_path / "bootstrap.sh"
    script.write_text(rendered, encoding="utf-8")

    completed = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f'readonly repo_url="{REPO_URL}"' in rendered
    assert f'readonly repo_commit="{DUMMY_COMMIT}"' in rendered
    assert "${HOME}/.local/share/acs-nemoclaw-launchable" in rendered
    assert "install -d -m 700" in rendered
    assert "git clone --quiet --no-checkout" in rendered
    assert "remote get-url --all origin" in rendered
    assert "checkout --quiet --detach" in rendered
    assert "symbolic-ref --quiet HEAD" in rendered
    assert "rev-parse --verify HEAD" in rendered
    assert "status --porcelain=v1 --untracked-files=all" in rendered
    assert rendered.index(
        "status --porcelain=v1 --untracked-files=all"
    ) < rendered.index('export NVIDIA_INFERENCE_API_KEY="${launch_key}"')
    assert (
        'exec /bin/bash "${checkout_dir}/launchable/acs_nemoclaw_launchable_setup.sh"'
        in rendered
    )


def test_console_bootstrap_hides_the_key_from_setup_children_then_executes_with_it(
    tmp_path: Path,
) -> None:
    canary = "inference-hub-console-bootstrap-canary"
    parent_key_canary = "nvapi-parent-env-must-not-win"
    parent_compatible_canary = "parent-compatible-key-must-not-win"
    rendered = _rendered_bootstrap(canary)
    script = tmp_path / "bootstrap.sh"
    script.write_text(rendered, encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_log = tmp_path / "children.log"
    final_key = tmp_path / "final-key"

    fake_install = fake_bin / "install"
    fake_install.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'install:%s:%s:%s\\n\' "${NVIDIA_INFERENCE_API_KEY-unset}" "${launch_key-unset}" "${COMPATIBLE_API_KEY-unset}" >> "${ACS_CHILD_LOG}"\n'
        'destination="${!#}"\n'
        'mkdir -p -- "${destination}"\n'
        'chmod 700 "${destination}"\n',
        encoding="utf-8",
    )
    fake_install.chmod(fake_install.stat().st_mode | stat.S_IXUSR)

    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        'printf \'git:%s:%s:%s:%s\\n\' "${NVIDIA_INFERENCE_API_KEY-unset}" "${launch_key-unset}" "${COMPATIBLE_API_KEY-unset}" "$*" >> "${ACS_CHILD_LOG}"\n'
        'if [[ "${1:-}" == clone ]]; then\n'
        '  destination="${!#}"\n'
        '  mkdir -p -- "${destination}/.git" "${destination}/launchable"\n'
        "  cat > \"${destination}/launchable/acs_nemoclaw_launchable_setup.sh\" <<'EOF'\n"
        "#!/usr/bin/env bash\n"
        'printf \'setup:%s:%s:%s\\n\' "${NVIDIA_INFERENCE_API_KEY-unset}" "${launch_key-unset}" "${COMPATIBLE_API_KEY-unset}" >> "${ACS_CHILD_LOG}"\n'
        'printf \'%s\' "${NVIDIA_INFERENCE_API_KEY:-}" > "${ACS_FINAL_KEY}"\n'
        "EOF\n"
        "  exit 0\n"
        "fi\n"
        'command_name="${3:-}"\n'
        'case "${command_name}" in\n'
        "  remote) printf '%s\\n' \"${ACS_REPO_URL}\" ;;\n"
        "  symbolic-ref) exit 1 ;;\n"
        "  rev-parse) printf '%s\\n' \"${ACS_COMMIT}\" ;;\n"
        "  fetch|cat-file|checkout|status) exit 0 ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment.update(
        {
            "ACS_CHILD_LOG": str(child_log),
            "ACS_COMMIT": DUMMY_COMMIT,
            "ACS_FINAL_KEY": str(final_key),
            "ACS_REPO_URL": REPO_URL,
            "COMPATIBLE_API_KEY": parent_compatible_canary,
            "HOME": str(tmp_path / "home"),
            "NVIDIA_INFERENCE_API_KEY": parent_key_canary,
            "launch_key": "nvapi-parent-launch-key-must-not-win",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert canary not in completed.stdout + completed.stderr
    assert parent_key_canary not in completed.stdout + completed.stderr
    assert parent_compatible_canary not in completed.stdout + completed.stderr
    assert final_key.read_text(encoding="utf-8") == canary
    assert parent_key_canary not in final_key.read_text(encoding="utf-8")
    child_records = child_log.read_text(encoding="utf-8").splitlines()
    assert child_records
    assert parent_key_canary not in "\n".join(child_records)
    assert parent_compatible_canary not in "\n".join(child_records)
    assert all(
        record.startswith(
            (
                "git:unset:unset:unset:",
                "install:unset:unset:unset",
                f"setup:{canary}:unset:unset",
            )
        )
        for record in child_records
    )
    checkout = (
        tmp_path
        / "home/.local/share/acs-nemoclaw-launchable"
        / f"source-{DUMMY_COMMIT}"
    )
    assert stat.S_IMODE(checkout.stat().st_mode) == 0o700


def test_console_bootstrap_rejects_a_dirty_reused_checkout_before_exporting_key(
    tmp_path: Path,
) -> None:
    canary = "nvapi-dirty-checkout-canary"
    rendered = _rendered_bootstrap(canary)
    script = tmp_path / "bootstrap.sh"
    script.write_text(rendered, encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    checkout = (
        tmp_path
        / "home/.local/share/acs-nemoclaw-launchable"
        / f"source-{DUMMY_COMMIT}"
    )
    (checkout / ".git").mkdir(parents=True)

    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        'command_name="${3:-}"\n'
        'case "${command_name}" in\n'
        "  remote) printf '%s\\n' \"${ACS_REPO_URL}\" ;;\n"
        "  status) printf ' M launchable/acs_nemoclaw_launchable_setup.sh\\n' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_REPO_URL": REPO_URL,
            "HOME": str(tmp_path / "home"),
            "NVIDIA_INFERENCE_API_KEY": canary,
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "not clean" in completed.stderr
    assert canary not in completed.stdout + completed.stderr


def test_console_bootstrap_fails_closed_when_git_status_fails(tmp_path: Path) -> None:
    canary = "nvapi-status-failure-canary"
    rendered = _rendered_bootstrap(canary)
    script = tmp_path / "bootstrap.sh"
    script.write_text(rendered, encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    checkout = (
        tmp_path
        / "home/.local/share/acs-nemoclaw-launchable"
        / f"source-{DUMMY_COMMIT}"
    )
    (checkout / ".git").mkdir(parents=True)

    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        'command_name="${3:-}"\n'
        'case "${command_name}" in\n'
        "  remote) printf '%s\\n' \"${ACS_REPO_URL}\" ;;\n"
        "  status) exit 9 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_REPO_URL": REPO_URL,
            "HOME": str(tmp_path / "home"),
            "NVIDIA_INFERENCE_API_KEY": canary,
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "could not verify" in completed.stderr
    assert canary not in completed.stdout + completed.stderr


def test_authoring_sheet_has_expected_definition_sections() -> None:
    source = AUTHORING_SHEET.read_text(encoding="utf-8")

    new_definition = "## Create a new definition"
    existing_definition = "## Update the existing saved definition"
    assert new_definition in source
    assert existing_definition in source
    assert source.index(new_definition) < source.index(existing_definition)
    assert "The fields below apply only to a new definition." in source


def _renderer_module():
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("launchable.render_acs_console_bootstrap")


def test_public_template_is_unrendered_and_fails_before_git(tmp_path: Path) -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert source.count(SENTINEL) == 1
    assert "set +x +v" in source
    assert "launch_key=__NVIDIA_INFERENCE_API_KEY__" in source
    assert "unset NVIDIA_INFERENCE_API_KEY NVIDIA_API_KEY COMPATIBLE_API_KEY" in source
    assert source.index("unrendered") < source.index("git clone")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_called = tmp_path / "git-called"
    (fake_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        f"touch {shlex.quote(str(git_called))}\n"
        "exit 99\n",
        encoding="utf-8",
    )
    (fake_bin / "git").chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "NVIDIA_INFERENCE_API_KEY": "nvapi-must-not-be-used",
            "NVIDIA_API_KEY": "nvapi-also-must-not-be-used",
        }
    )
    completed = subprocess.run(
        ["bash", str(PUBLIC_BOOTSTRAP)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "unrendered" in completed.stderr
    assert not git_called.exists()
    assert "nvapi-must-not-be-used" not in completed.stdout + completed.stderr


def test_renderer_validates_and_shell_quotes_api_key() -> None:
    renderer = _renderer_module()
    template = "launch_key=__NVIDIA_INFERENCE_API_KEY__\n"
    key = "inference-hub-a'b$"
    rendered = renderer.render_acs_console_bootstrap(template, key)
    assert rendered == f'launch_key={shlex.quote(key)}\n'
    assert rendered.count(SENTINEL) == 0
    for malformed_template in (
        "launch_key=plain\n",
        "launch_key=__NVIDIA_INFERENCE_API_KEY____NVIDIA_INFERENCE_API_KEY__\n",
    ):
        try:
            renderer.render_acs_console_bootstrap(malformed_template, key)
        except ValueError:
            pass
        else:
            raise AssertionError("renderer accepted a malformed sentinel count")
    for invalid in (
        "",
        " ",
        "\t",
        "hub key",
        SENTINEL,
        "hub-a\x07b",
        "hub-a\x7fb",
        "hub-a\n b",
        "hub-a\rb",
        "hub-a\x00b",
    ):
        try:
            renderer.render_acs_console_bootstrap(template, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"renderer accepted invalid key: {invalid!r}")


def test_renderer_writes_new_private_output_without_leaking_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    renderer = _renderer_module()
    template = "launch_key=__NVIDIA_INFERENCE_API_KEY__\n"
    monkeypatch.setattr(renderer, "TEMPLATE_PATH", tmp_path / "template.sh")
    renderer.TEMPLATE_PATH.write_text(template, encoding="utf-8")
    key = "inference-hub-output-canary"
    monkeypatch.setattr(renderer.getpass, "getpass", lambda prompt: key)
    output = tmp_path / "private.sh"
    assert renderer.main([str(output)]) == 0
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert output.read_text(encoding="utf-8") == f"launch_key={shlex.quote(key)}\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.stat().st_uid == os.getuid()
    try:
        renderer.main([str(output)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("renderer overwrote an existing output")


def test_renderer_default_input_is_the_pinned_public_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:
    renderer = _renderer_module()
    key = "inference-hub-default-template-canary"
    monkeypatch.setattr(renderer.getpass, "getpass", lambda prompt: key)
    output = tmp_path / "private.sh"
    assert renderer.main([str(output)]) == 0
    rendered = output.read_text(encoding="utf-8")
    assert f'readonly repo_commit="{PINNED_SETUP_COMMIT}"' in rendered
    assert PLACEHOLDER not in rendered
    assert SENTINEL not in rendered
    completed = subprocess.run(
        ["bash", "-n", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_renderer_malformed_cli_input_leaves_no_output_and_does_not_print_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    renderer = _renderer_module()
    key = " \t "
    monkeypatch.setattr(renderer.getpass, "getpass", lambda prompt: key)
    output = tmp_path / "private.sh"
    try:
        renderer.main([str(output)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("renderer accepted malformed CLI input")
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert not output.exists()


def test_rendered_bootstrap_has_separate_unrendered_and_malformed_key_failures(
    tmp_path: Path,
) -> None:
    for key_text, expected_error in (
        (SENTINEL, "unrendered"),
        ("", "must not be empty or contain whitespace"),
        ("   ", "must not be empty or contain whitespace"),
        ("hub key", "must not be empty or contain whitespace"),
        ("hub-a\x07b", "must not contain control characters"),
        ("hub-a\x7fb", "must not contain control characters"),
    ):
        rendered = _rendered_bootstrap(key_text)
        script = tmp_path / f"bootstrap-{expected_error}-{len(key_text)}.sh"
        script.write_text(rendered, encoding="utf-8")
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            env={**os.environ, "launch_key": "parent-key-canary"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert expected_error in completed.stderr
        assert "parent-key-canary" not in completed.stdout + completed.stderr


def test_renderer_rejects_repo_and_symlinked_parent_paths(tmp_path: Path) -> None:
    renderer = _renderer_module()
    outside = tmp_path / "outside"
    outside.mkdir()
    for candidate in (ROOT / "private.sh", tmp_path / "link" / "private.sh"):
        if candidate.parent.name == "link":
            candidate.parent.symlink_to(outside, target_is_directory=True)
        try:
            renderer._validate_output_path(str(candidate))
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError(f"renderer accepted unsafe output path: {candidate}")


def test_renderer_cleans_partial_output_when_parent_identity_changes(tmp_path: Path) -> None:
    renderer = _renderer_module()
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "private.sh"
    expected_parent = os.lstat(parent)
    moved = tmp_path / "moved"
    parent.rename(moved)
    parent.mkdir()
    try:
        renderer._write_private_output(output, expected_parent, "private")
    except OSError:
        pass
    else:
        raise AssertionError("renderer ignored a changed output parent")
    assert not output.exists()


def test_renderer_removes_partial_output_when_write_fails(tmp_path: Path, monkeypatch) -> None:
    renderer = _renderer_module()
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "private.sh"
    expected_parent = os.lstat(parent)

    def fail_open(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(renderer.os, "fdopen", fail_open)
    try:
        renderer._write_private_output(output, expected_parent, "private")
    except OSError:
        pass
    else:
        raise AssertionError("renderer ignored a write failure")
    assert not output.exists()


def test_authoring_sheet_uses_the_bootstrap_as_the_only_setup_source() -> None:
    source = AUTHORING_SHEET.read_text(encoding="utf-8")
    assert "The fields below apply only to a new definition." in source
    assert "Do not select **Create Launchable** for this update." in source
    assert "I don’t have any code files" in source
    assert "launchable/acs_console_bootstrap.sh.in" in source
    assert "Do not paste the public bootstrap" in source
    assert "Do not paste `launchable/acs_nemoclaw_launchable_setup.sh`" in source
    assert "four fixed prompts" in source
    assert "canonical `answer_markdown`" in source
    assert "source push alone does not update" in source
    in_place = (
        "Validate the updated source in place on the task-owned existing instance "
        "before publication."
    )
    source_boundary = (
        "This in-place pass validates the source and runtime, not the saved "
        "Launchable bootstrap."
    )
    publish_source = (
        "After that in-place pass succeeds, publish the reviewed source commit."
    )
    assert in_place in source
    assert source_boundary in source
    assert publish_source in source
    assert "Commit and push the generated bootstrap second." in source
    private_bootstrap = (
        "Save only the private rendered body in the setup-script field of the "
        "existing saved Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`."
    )
    assert private_bootstrap in source
    assert "Keep access set to **Only my organization** during this update." in source
    fresh_deployment = "Create a future fresh deployment from the saved definition."
    broader_access = (
        "Only after this fresh-deployment pass may access change to "
        "**Anyone with the link**."
    )
    assert fresh_deployment in source
    assert "automatic OpenClaw sign-in" in source
    assert "all four prompts" in source
    assert "all four images" in source
    assert "Download Results" in source
    assert broader_access in source
    sequence = tuple(
        source.index(item)
        for item in (
            in_place,
            source_boundary,
            publish_source,
            "Commit and push the generated bootstrap second.",
            private_bootstrap,
            fresh_deployment,
            broader_access,
        )
    )
    assert sequence == tuple(sorted(sequence))
    assert "source and bootstrap commits pass live acceptance" not in source
    assert "`NVIDIA_INFERENCE_API_KEY`" not in source
    assert "Never store the key in source" in source
    assert "Launchable default" in source
    assert "documentation" in source
    assert "logs" in source
    assert "16,384 bytes" in source
    assert "| `Open Chemistry Agent` | `18788` |" in source
    assert "| `Download Results` | `8765` |" in source
    assert "Do not add a Secure Link for port `18789`." in source
    assert "Do not expose raw TCP or UDP ports." in source
    assert "one time-bounded agent turn" not in source
    assert "edit a bounded chemistry task" not in source
    assert "acs_task_prompt.txt" not in source


def test_authoring_sheet_has_the_zero_input_private_render_workflow() -> None:
    source = AUTHORING_SHEET.read_text(encoding="utf-8")
    create_start = source.index("## Create a new definition")
    update_start = source.index("## Update the existing saved definition")
    create_section = source[create_start:update_start]
    update_section = source[update_start:]
    render_command = (
        "`python3 launchable/render_acs_console_bootstrap.py "
        "/private/tmp/acs-openclaw-workshop-setup.sh`"
    )

    for section in (create_section, update_section):
        assert "The OpenClaw Launchable has no Launch parameters or Setup values." in section
        assert render_command in section
        assert "workshop-only `inference.nvidia.com` API key only at the hidden prompt" in section
        assert "owner, regular-file type, mode `0600`" in section
        assert "Bash syntax" in section
        assert "byte size" in section
        assert "without printing its contents" in section
        assert "private rendered body" in section
        assert "After the save is confirmed, delete" in section
        assert "Never store the key in source" in section
        assert "Launchable default" in section
        assert "documentation" in section
        assert "logs" in section
        assert "Monitor its use during the workshop" in section
        assert "revoke it after the workshop" in section

    assert (
        "Save only the private rendered body in the setup-script field of the "
        "existing saved Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`."
        in update_section
    )
    assert source.count("| `Open Chemistry Agent` | `18788` |") == 1
    assert source.count("| `Download Results` | `8765` |") == 1
