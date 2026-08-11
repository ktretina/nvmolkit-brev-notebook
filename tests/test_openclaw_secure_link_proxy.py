from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_TEST = ROOT / "tests" / "openclaw_secure_link_proxy.test.mjs"


def test_openclaw_secure_link_proxy_node_contract() -> None:
    completed = subprocess.run(
        ["node", "--test", str(NODE_TEST)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
