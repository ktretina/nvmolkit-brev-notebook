import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launchable" / "nemoclaw_phase_zero.sh"
PYTORCH_POLICY = ROOT / "launchable" / "pytorch-cu128-policy.yaml"
NVMOLKIT_INSTALLER = ROOT / "launchable" / "install_nvmolkit_in_sandbox.sh"
GPU_PROBE = ROOT / "launchable" / "nvmolkit_gpu_probe.py"
WORKSPACE_TOOLS = ROOT / "launchable" / "acs_workspace_tools.md"
ARTIFACT_SERVER = ROOT / "launchable" / "start_artifact_server.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_detached_phase_zero(
    tmp_path: Path,
    *,
    key_value: str = "inference-hub-test-secret",
    key_payload: bytes | None = None,
    installer_exit: int = 1,
    failure_step: str = "provider_selection",
    interrupted: bool = True,
    install_nemoclaw: bool = True,
    resumable: bool = True,
    resume_exit: int = 0,
    session_is_object: bool = True,
    symlink_session: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    state_dir = tmp_path / "state"
    fake_bin.mkdir()
    (home / ".config" / "acs-phase-zero").mkdir(parents=True)

    key_file = home / ".config" / "acs-phase-zero" / "NVIDIA_INFERENCE_API_KEY"
    if key_payload is None:
        key_file.write_text(f"{key_value}\n")
    else:
        key_file.write_bytes(key_payload)
    key_file.chmod(0o600)

    resume_log = tmp_path / "resume.log"
    openshell_log = tmp_path / "openshell.log"
    fake_installer = tmp_path / "installer.sh"
    fake_session = tmp_path / "onboard-session.json"
    fake_nemoclaw = tmp_path / "nemoclaw"
    fake_openshell = tmp_path / "openshell"

    if session_is_object:
        fake_session.write_text(
            "{\n"
            '  "status": "failed",\n'
            f'  "resumable": {str(resumable).lower()},\n'
            '  "failure": {\n'
            f'    "step": "{failure_step}",\n'
            f'    "interrupted": {str(interrupted).lower()}\n'
            "  }\n"
            "}\n"
        )
    else:
        fake_session.write_text("[]\n")
    _write_executable(
        fake_nemoclaw,
        "#!/usr/bin/env bash\n"
        '[[ "${COMPATIBLE_API_KEY:-}" == "inference-hub-test-secret" ]] || exit 91\n'
        '[[ -z "${NVIDIA_INFERENCE_API_KEY:-}" ]] || exit 93\n'
        '[[ "${NEMOCLAW_ONBOARD_VALIDATION_TIMEOUT_SECONDS:-}" == 60 ]] || exit 92\n'
        'printf \'%s\\n\' "$*" >> "$RESUME_LOG"\n'
        'openshell "$@"\n',
    )
    _write_executable(
        fake_openshell,
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$OPENSHELL_LOG"\n'
        'exit "$RESUME_EXIT"\n',
    )
    _write_executable(
        fake_installer,
        "#!/usr/bin/env bash\n"
        '[[ "${COMPATIBLE_API_KEY:-}" == "inference-hub-test-secret" ]] || exit 81\n'
        '[[ -z "${NVIDIA_INFERENCE_API_KEY:-}" ]] || exit 82\n'
        '[[ "${NEMOCLAW_PROVIDER:-}" == custom ]] || exit 83\n'
        '[[ "${NEMOCLAW_ENDPOINT_URL:-}" == https://inference-api.nvidia.com/v1 ]] || exit 84\n'
        '[[ "${NEMOCLAW_MODEL:-}" == nvidia/nvidia/nemotron-3-super-v3 ]] || exit 85\n'
        '[[ "${NEMOCLAW_PREFERRED_API:-}" == openai-completions ]] || exit 86\n'
        '/bin/mkdir -p "$HOME/.local/bin" "$HOME/.nemoclaw"\n'
        'if [[ "$INSTALL_NEMOCLAW" == 1 ]]; then\n'
        '  /bin/cp "$FAKE_NEMOCLAW" "$HOME/.local/bin/nemoclaw"\n'
        '  /bin/chmod 700 "$HOME/.local/bin/nemoclaw"\n'
        "fi\n"
        '/bin/cp "$FAKE_OPENSHELL" "$HOME/.local/bin/openshell"\n'
        '/bin/chmod 700 "$HOME/.local/bin/openshell"\n'
        'if [[ "$SYMLINK_SESSION" == 1 ]]; then\n'
        '  /bin/ln -s "$FAKE_SESSION" "$HOME/.nemoclaw/onboard-session.json"\n'
        "else\n"
        '  /bin/cp "$FAKE_SESSION" "$HOME/.nemoclaw/onboard-session.json"\n'
        "fi\n"
        'exit "$INSTALLER_EXIT"\n',
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        '[[ -z "${COMPATIBLE_API_KEY:-}" && -z "${NVIDIA_INFERENCE_API_KEY:-}" ]] || exit 71\n'
        "while (( $# )); do\n"
        '  if [[ "$1" == -o ]]; then /bin/cp "$FAKE_INSTALLER" "$2"; exit 0; fi\n'
        "  shift\n"
        "done\n"
        "exit 2\n",
    )
    _write_executable(
        fake_bin / "sha256sum",
        "#!/usr/bin/env bash\n"
        '[[ -z "${COMPATIBLE_API_KEY:-}" && -z "${NVIDIA_INFERENCE_API_KEY:-}" ]] || exit 72\n'
        "/bin/cat >/dev/null\n",
    )
    _write_executable(
        fake_bin / "stat",
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == -c && "$2" == %a ]]; then\n'
        '  /usr/bin/stat -f %Lp "$3"\n'
        "else\n"
        '  /usr/bin/stat "$@"\n'
        "fi\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "ACS_PHASE_ZERO_DETACHED": "1",
            "ACS_PHASE_ZERO_KEY_FILE": str(key_file),
            "ACS_PHASE_ZERO_STATE_DIR": str(state_dir),
            "COMPATIBLE_API_KEY": "parent-compatible-key-must-not-reach-children",
            "FAKE_INSTALLER": str(fake_installer),
            "FAKE_NEMOCLAW": str(fake_nemoclaw),
            "FAKE_OPENSHELL": str(fake_openshell),
            "FAKE_SESSION": str(fake_session),
            "HOME": str(home),
            "INSTALLER_EXIT": str(installer_exit),
            "INSTALL_NEMOCLAW": "1" if install_nemoclaw else "0",
            "NVIDIA_INFERENCE_API_KEY": "parent-nvidia-key-must-not-reach-children",
            "OPENSHELL_LOG": str(openshell_log),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "RESUME_LOG": str(resume_log),
            "RESUME_EXIT": str(resume_exit),
            "SYMLINK_SESSION": "1" if symlink_session else "0",
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, state_dir, resume_log


def test_phase_zero_setup_is_pinned_and_removes_transport_key_before_install():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = SCRIPT.read_text()
    assert "0d1cb93888c817daec44b2cc996afa75eebcbd46" in source
    assert "b52f053a550fab90ab1dff4ab7f3a0b55b2506aeafd2062832e65632fdbcae70" in source
    assert "NEMOCLAW_PROVIDER=custom" in source
    assert "NEMOCLAW_ENDPOINT_URL=https://inference-api.nvidia.com/v1" in source
    assert "NEMOCLAW_MODEL=nvidia/nvidia/nemotron-3-super-v3" in source
    assert "NEMOCLAW_PREFERRED_API=openai-completions" in source
    assert "NEMOCLAW_PROVIDER=build" not in source
    assert "NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS" not in source
    assert "NEMOCLAW_E2E_USE_HOSTED_INFERENCE" not in source
    assert "NEMOCLAW_SANDBOX_GPU=1" in source
    assert "export COMPATIBLE_API_KEY" in source
    assert "export NVIDIA_INFERENCE_API_KEY" not in source
    assert source.index("unset COMPATIBLE_API_KEY NVIDIA_INFERENCE_API_KEY") < source.index(
        'curl -fsSL "${install_url}"'
    )
    assert source.index('rm -f -- "${key_file}"') < source.index('bash "${installer}"')


def test_phase_zero_setup_detaches_before_reading_the_key(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux_log = tmp_path / "tmux.log"
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$TMUX_LOG"\n'
        'if [[ "${1:-}" == has-session ]]; then exit 1; fi\n'
    )
    fake_tmux.chmod(fake_tmux.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TMUX_LOG": str(tmux_log),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = tmux_log.read_text()
    assert "new-session -d -s acs-phase-zero-install" in commands
    assert "send-keys -t acs-phase-zero-install" in commands
    assert "NemoClaw installation started" in completed.stdout


def test_phase_zero_resumes_one_interrupted_provider_selection_failure(tmp_path):
    completed, state_dir, resume_log = _run_detached_phase_zero(tmp_path)
    key_file = (
        tmp_path / "home" / ".config" / "acs-phase-zero" / "NVIDIA_INFERENCE_API_KEY"
    )

    assert completed.returncode == 0, completed.stderr
    assert not key_file.exists()
    assert "inference-hub-test-secret" not in completed.stdout
    assert "inference-hub-test-secret" not in completed.stderr
    assert (state_dir / "install.first.exit").read_text() == "1\n"
    assert (state_dir / "install.resume.exit").read_text() == "0\n"
    assert (state_dir / "install.exit").read_text() == "0\n"
    assert stat.S_IMODE((state_dir / "install.first.exit").stat().st_mode) == 0o600
    assert stat.S_IMODE((state_dir / "install.resume.exit").stat().st_mode) == 0o600
    assert resume_log.read_text() == (
        "onboard --resume --non-interactive --yes --yes-i-accept-third-party-software\n"
    )


def test_phase_zero_preserves_the_single_resume_failure_as_final_status(tmp_path):
    completed, state_dir, resume_log = _run_detached_phase_zero(
        tmp_path,
        resume_exit=17,
    )
    key_file = (
        tmp_path / "home" / ".config" / "acs-phase-zero" / "NVIDIA_INFERENCE_API_KEY"
    )

    assert completed.returncode == 17
    assert not key_file.exists()
    assert "inference-hub-test-secret" not in completed.stdout
    assert "inference-hub-test-secret" not in completed.stderr
    assert (state_dir / "install.first.exit").read_text() == "1\n"
    assert (state_dir / "install.resume.exit").read_text() == "17\n"
    assert (state_dir / "install.exit").read_text() == "17\n"
    assert resume_log.read_text().count("onboard --resume") == 1


def test_phase_zero_does_not_resume_a_nonstandard_installer_failure(tmp_path):
    completed, state_dir, resume_log = _run_detached_phase_zero(
        tmp_path,
        installer_exit=2,
    )

    assert completed.returncode == 2
    assert (state_dir / "install.first.exit").read_text() == "2\n"
    assert not (state_dir / "install.resume.exit").exists()
    assert (state_dir / "install.exit").read_text() == "2\n"
    assert not resume_log.exists()


def test_phase_zero_rejects_empty_or_whitespace_inference_hub_keys(tmp_path):
    for case, key_value in (("empty", "   "), ("embedded", "hub key")):
        case_dir = tmp_path / case
        case_dir.mkdir()
        completed, state_dir, resume_log = _run_detached_phase_zero(
            case_dir,
            key_value=key_value,
        )

        assert completed.returncode == 2
        assert "Protected NVIDIA inference key is malformed." in completed.stderr
        assert key_value not in completed.stdout
        assert not (state_dir / "install.first.exit").exists()
        assert (state_dir / "install.exit").read_text() == "2\n"
        assert not resume_log.exists()


def test_phase_zero_rejects_control_multiline_nul_and_sentinel_keys(tmp_path):
    malformed_payloads = (
        ("empty", b"\n"),
        ("bel", b"hub\x07key\n"),
        ("del", b"hub\x7fkey\n"),
        ("carriage-return", b"hub\rkey\n"),
        ("multiline", b"hub-key\nsecond-line\n"),
        ("nul", b"hub\x00key\n"),
        ("sentinel", b"__NVIDIA_INFERENCE_API_KEY__\n"),
    )
    for case, key_payload in malformed_payloads:
        case_dir = tmp_path / case
        case_dir.mkdir()
        completed, state_dir, resume_log = _run_detached_phase_zero(
            case_dir,
            key_payload=key_payload,
        )

        assert completed.returncode == 2, case
        assert "Protected NVIDIA inference key is malformed." in completed.stderr, case
        assert not (state_dir / "install.first.exit").exists(), case
        assert (state_dir / "install.exit").read_text() == "2\n", case
        assert not resume_log.exists(), case


def test_phase_zero_does_not_resume_an_unsafe_or_unrelated_failure(tmp_path):
    for case, options in (
        ("unsafe-session", {"symlink_session": True}),
        ("wrong-step", {"failure_step": "gateway"}),
        ("not-interrupted", {"interrupted": False}),
        ("not-resumable", {"resumable": False}),
        ("non-object-session", {"session_is_object": False}),
        ("missing-cli", {"install_nemoclaw": False}),
    ):
        case_dir = tmp_path / case
        case_dir.mkdir()
        completed, state_dir, resume_log = _run_detached_phase_zero(
            case_dir,
            **options,
        )

        assert completed.returncode == 1, case
        assert (state_dir / "install.first.exit").read_text() == "1\n", case
        assert not (state_dir / "install.resume.exit").exists(), case
        assert (state_dir / "install.exit").read_text() == "1\n", case
        assert not resume_log.exists(), case
        assert completed.stderr == "", case


def test_pytorch_policy_is_narrow_and_read_only():
    source = PYTORCH_POLICY.read_text()

    assert "host: download.pytorch.org" in source
    assert "host: download-r2.pytorch.org" in source
    assert "host: pypi.nvidia.com" in source
    assert "port: 443" in source
    assert "protocol: rest" in source
    assert "enforcement: enforce" in source
    assert "method: GET" in source
    assert "method: HEAD" in source
    assert "path: /usr/bin/python3*" in source
    assert "method: POST" not in source
    assert source.count("host:") == 3


def test_sandbox_installer_uses_only_the_minimal_pinned_chemistry_stack():
    completed = subprocess.run(
        ["bash", "-n", str(NVMOLKIT_INSTALLER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = NVMOLKIT_INSTALLER.read_text()
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "torch==2.7.1+cu128" in source
    assert "nvmolkit==0.5.0" in source
    assert "pandas==2.3.1" in source
    assert "matplotlib==3.10.3" in source
    assert "install.exit" in source
    assert "jupyter" not in source.lower()
    assert "seaborn" not in source.lower()


def test_gpu_probe_exercises_real_morgan_fingerprints_on_cuda():
    source = GPU_PROBE.read_text()
    compile(source, str(GPU_PROBE), "exec")

    assert "MorganFingerprintGenerator" in source
    assert "GetFingerprints" in source
    assert "torch.cuda.synchronize()" in source
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.get_device_name(0)" in source
    assert "(3, 32)" in source


def test_workspace_note_exposes_only_the_bounded_workshop_commands_and_artifacts():
    source = WORKSPACE_TOOLS.read_text()

    assert "nvmolkit-usage" in source
    assert "remains available after the bounded exercise" in source
    assert "Read the installed" not in source
    assert "Do not read files" in source
    runner = (
        "env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 "
        "/sandbox/.openclaw/workspace/acs_workshop_runner.py"
    )
    lessons = (
        "data-and-representation",
        "relationships-and-groups",
        "sampled-3d-geometry",
    )
    assert source.count("run-lesson") == 3
    for lesson in lessons:
        assert f"{runner} run-lesson {lesson}" in source
    assert source.count(f"{runner} objective-start") == 1
    assert source.count(f"{runner} objective-step") == 1
    assert "--state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'" in source
    for path in (
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/01-inspection/library_preview.png",
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png",
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/06-mmff94/optimized_structures.png",
        "MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png",
        "workshop/results.zip",
    ):
        assert path in source
    assert "Do not install" in source
    assert "Do not use the network" in source
    assert "Do not edit" in source
    assert ".acs-workshop-state" in source
    assert "run-stage" not in source
    assert "<script.py>" not in source
    assert "not evidence of biological activity" in source


def test_workspace_note_stops_completed_lessons_and_copies_canonical_answers():
    source = WORKSPACE_TOOLS.read_text()
    assert "Each lesson command has a one-call budget." in source
    assert "top-level `status: complete`" in source
    assert "stop tool use and answer from that first result" in source
    assert "Never run that lesson again in the same prompt." in source
    assert "An empty assistant response does not permit another tool call." in source
    assert "Copy the decoded `answer_markdown` string exactly" in source
    assert "returns both conformer stages; run it only once" in source
    assert "exact returned `state_id` and `swap_id`" in source
    assert "Keep both values single-quoted" in source
    assert "Run at most three objective-step commands" in source


def test_workspace_note_preserves_scientific_boundaries():
    source = WORKSPACE_TOOLS.read_text()
    assert "deterministic 256-record ChEMBL convenience sample" in source
    assert "non-representative chemical space" in source
    assert "radius-2, 1024-bit hashed fingerprint" in source
    assert "real GPU execution with no acceleration or speedup claim" in source
    assert "cutoff `0.40` is Tanimoto distance" in source
    assert "similarity `1.0` does not prove molecular identity" in source
    assert (
        "nvMolKit computes fingerprints and Tanimoto similarities on GPU; "
        "RDKit runs Butina clustering on CPU." in source
    )
    assert "`D_min` is the minimum pairwise Tanimoto distance" in source
    assert "`D_min = min(1 - Tanimoto similarity)`" in source
    assert "higher `D_min` means greater separation" in source
    assert "weakest-link diversity score within eight fixed candidates" in source
    assert "Never call `D_min` a similarity score." in source
    assert (
        "deterministic selected molecules are not centroids, medoids, or "
        "globally optimal representatives" in source
    )
    assert (
        "Do not report intermediate, predicted, target, or per-step scores" in source
    )
    assert (
        "structural-descriptor objective does not demonstrate unrestricted "
        "autonomous design or biological performance" in source
    )


def test_artifact_server_is_retry_safe_and_serves_only_outputs():
    completed = subprocess.run(
        ["bash", "-n", str(ARTIFACT_SERVER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = ARTIFACT_SERVER.read_text()
    assert "tmux has-session -t acs-artifacts" in source
    assert "python3 -m http.server 8765" in source
    assert "--bind 0.0.0.0" in source
    assert "--directory /sandbox/.openclaw/workspace/outputs" in source
