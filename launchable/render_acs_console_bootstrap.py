#!/usr/bin/env python3.12
"""Render a private ACS Console bootstrap without exposing its credential."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import stat
from pathlib import Path


SENTINEL = "__NVIDIA_INFERENCE_API_KEY__"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEMPLATE_PATH = SCRIPT_DIR / "acs_console_bootstrap.sh"


def render_acs_console_bootstrap(template_text: str, key: str) -> str:
    """Return the template with one validated key encoded as a shell word."""
    if template_text.count(SENTINEL) != 1:
        raise ValueError("ACS bootstrap template must contain exactly one credential sentinel")
    if any(character.isspace() for character in key):
        raise ValueError("credential must not contain whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in key):
        raise ValueError("credential must not contain control characters")
    if not key or key == SENTINEL:
        raise ValueError("credential must not be empty or unrendered")
    return template_text.replace(SENTINEL, shlex.quote(key), 1)


def _validate_output_path(value: str) -> tuple[Path, os.stat_result]:
    output_path = Path(os.path.abspath(Path(value).expanduser()))
    if os.path.lexists(output_path):
        raise FileExistsError("output path already exists")
    output_parent = output_path.parent
    try:
        parent_status = os.lstat(output_parent)
    except FileNotFoundError as error:
        raise ValueError("output parent must already exist") from error
    if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
        raise ValueError("output parent must be a real directory, not a symlink")
    physical_parent = output_parent.resolve(strict=True)
    for ancestor in (
        output_parent,
        *output_parent.parents,
        physical_parent,
        *physical_parent.parents,
    ):
        if os.path.samefile(ancestor, REPO_ROOT):
            raise ValueError("output path must be outside the repository")
    return output_path, parent_status


def _write_private_output(
    output_path: Path,
    expected_parent: os.stat_result,
    rendered_text: str,
) -> None:
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(output_path.parent, parent_flags)
    try:
        opened_parent = os.fstat(parent_descriptor)
        expected_identity = (expected_parent.st_dev, expected_parent.st_ino)
        opened_identity = (opened_parent.st_dev, opened_parent.st_ino)
        if opened_identity != expected_identity or not stat.S_ISDIR(opened_parent.st_mode):
            raise OSError("output parent changed during rendering")

        output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_descriptor = -1
        created = False
        try:
            output_descriptor = os.open(
                output_path.name,
                output_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            created = True
            os.fchmod(output_descriptor, 0o600)
            output = os.fdopen(output_descriptor, "w", encoding="utf-8", newline="")
            output_descriptor = -1
            with output:
                output.write(rendered_text)
        except BaseException:
            if output_descriptor >= 0:
                os.close(output_descriptor)
            if created:
                try:
                    os.unlink(output_path.name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            raise
    finally:
        os.close(parent_descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a private ACS Console bootstrap outside this repository."
    )
    parser.add_argument("output", help="new output path, such as /tmp/acs-bootstrap-private.sh")
    args = parser.parse_args(argv)

    try:
        output_path, parent_status = _validate_output_path(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    key = getpass.getpass("inference.nvidia.com API key: ")
    try:
        rendered_text = render_acs_console_bootstrap(template_text, key)
        _write_private_output(output_path, parent_status, rendered_text)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(output_path)
    print("Keep this mode-0600 file private and use it only in the Brev Console.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
