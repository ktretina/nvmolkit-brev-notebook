#!/usr/bin/env python3.12
"""Render a private Brev setup script without exposing its credential."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
from pathlib import Path

SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEMPLATE_PATH = SCRIPT_DIR / "setup.sh"


def render_setup(template_text: str, key: str) -> str:
    """Return the template with one validated key encoded as a shell word."""
    if template_text.count(SENTINEL) != 1:
        raise ValueError("setup template must contain exactly one credential sentinel")
    if not key.startswith("sk-"):
        raise ValueError("credential must be an Inference Hub key beginning with sk-")
    if key == "sk-":
        raise ValueError("credential must not be empty after sk-")
    if any(character in key for character in ("\r", "\n", "\x00")):
        raise ValueError("credential must be one line without NUL")
    return template_text.replace(SENTINEL, shlex.quote(key), 1)


def _resolve_output_path(value: str) -> Path:
    output_path = Path(value).expanduser().resolve()
    if output_path.is_relative_to(REPO_ROOT):
        raise ValueError("output path must be outside the repository")
    if output_path.exists():
        raise FileExistsError("output path already exists")
    return output_path


def _write_private_output(output_path: Path, rendered_text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = -1
        with output:
            output.write(rendered_text)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        output_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a private workshop setup script outside this repository."
    )
    parser.add_argument("output", help="new output path, such as /tmp/setup-private.sh")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args.output)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    key = getpass.getpass("NVIDIA Inference Hub key: ")
    try:
        rendered_text = render_setup(template_text, key)
        _write_private_output(output_path, rendered_text)
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))

    print(output_path)
    print("Keep this mode-0600 file private and use it only in the Brev Console.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
