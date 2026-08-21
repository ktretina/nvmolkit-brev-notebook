from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "launchable" / "acs_console_bootstrap.sh.in"
PUBLIC_BOOTSTRAP = ROOT / "launchable" / "acs_console_bootstrap.sh"
AUTHORING_SHEET = ROOT / "launchable" / "ACS_LAUNCHABLE_FIELDS.md"
PLACEHOLDER = "@REVIEWED_PUBLIC_COMMIT_SHA@"
REPO_URL = "https://github.com/ktretina/nvmolkit-brev-notebook.git"
DUMMY_COMMIT = "a" * 40
PINNED_SETUP_COMMIT = "1fa357b23e06d0eeb125519d55507f4104ed4594"


def _rendered_bootstrap() -> str:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert source.count(PLACEHOLDER) == 1
    return source.replace(PLACEHOLDER, DUMMY_COMMIT)


def test_public_console_bootstrap_is_exact_pinned_and_below_brev_limit() -> None:
    expected = TEMPLATE.read_text(encoding="utf-8").replace(
        PLACEHOLDER, PINNED_SETUP_COMMIT
    )
    published = PUBLIC_BOOTSTRAP.read_text(encoding="utf-8")

    assert published == expected
    assert published.startswith("#!/usr/bin/env bash\n")
    assert published.count(PINNED_SETUP_COMMIT) == 1
    assert PLACEHOLDER not in published
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
    rendered = _rendered_bootstrap()
    script = tmp_path / "bootstrap.sh"
    script.write_text(rendered, encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_log = tmp_path / "children.log"
    final_key = tmp_path / "final-key"

    fake_install = fake_bin / "install"
    fake_install.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'install:%s\\n\' "${NVIDIA_INFERENCE_API_KEY-unset}" >> "${ACS_CHILD_LOG}"\n'
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
        'printf \'git:%s:%s\\n\' "${NVIDIA_INFERENCE_API_KEY-unset}" "$*" >> "${ACS_CHILD_LOG}"\n'
        'if [[ "${1:-}" == clone ]]; then\n'
        '  destination="${!#}"\n'
        '  mkdir -p -- "${destination}/.git" "${destination}/launchable"\n'
        "  cat > \"${destination}/launchable/acs_nemoclaw_launchable_setup.sh\" <<'EOF'\n"
        "#!/usr/bin/env bash\n"
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

    canary = "nvapi-console-bootstrap-canary"
    environment = os.environ.copy()
    environment.update(
        {
            "ACS_CHILD_LOG": str(child_log),
            "ACS_COMMIT": DUMMY_COMMIT,
            "ACS_FINAL_KEY": str(final_key),
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

    assert completed.returncode == 0, completed.stderr
    assert canary not in completed.stdout + completed.stderr
    assert final_key.read_text(encoding="utf-8") == canary
    child_records = child_log.read_text(encoding="utf-8").splitlines()
    assert child_records
    assert all(
        record.startswith(("git:unset:", "install:unset")) for record in child_records
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
    rendered = _rendered_bootstrap()
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
    canary = "nvapi-dirty-checkout-canary"
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
    rendered = _rendered_bootstrap()
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
    canary = "nvapi-status-failure-canary"
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


def test_authoring_sheet_uses_the_bootstrap_as_the_only_setup_source() -> None:
    source = AUTHORING_SHEET.read_text(encoding="utf-8")

    new_definition = "## Create a new definition"
    existing_definition = "## Update the existing saved definition"
    assert new_definition in source
    assert existing_definition in source
    assert source.index(new_definition) < source.index(existing_definition)
    assert "The fields below apply only to a new definition." in source
    assert "Do not select **Create Launchable** for this update." in source
    assert "I don’t have any code files" in source
    assert "launchable/acs_console_bootstrap.sh.in" in source
    assert "Paste the bootstrap" in source
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
    generated_bootstrap = (
        "Paste the exact generated `launchable/acs_console_bootstrap.sh` into the "
        "existing saved Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`."
    )
    assert generated_bootstrap in source
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
            generated_bootstrap,
            fresh_deployment,
            broader_access,
        )
    )
    assert sequence == tuple(sorted(sequence))
    assert "source and bootstrap commits pass live acceptance" not in source
    assert "`NVIDIA_INFERENCE_API_KEY`" in source
    assert "Never put the API key in source" in source
    assert "Launchable default" in source
    assert "Enter it only in the private required Console parameter" in source
    assert "16,384 bytes" in source
    assert "| `Open Chemistry Agent` | `18788` |" in source
    assert "| `Download Results` | `8765` |" in source
    assert "Do not add a Secure Link for port `18789`." in source
    assert "Do not expose raw TCP or UDP ports." in source
    assert "one time-bounded agent turn" not in source
    assert "edit a bounded chemistry task" not in source
    assert "acs_task_prompt.txt" not in source
