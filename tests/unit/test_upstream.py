"""Verify upstream submodules are checked out and match pinned revisions."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

EXPECTED_SUBMODULES = {
    "lib/openspace": "11bdf128d9b53a4107aec1d3098fa04ab808a700",
    "lib/ai-trader": "3b3169b756002518b752baae994a2d1bdbb70600",
}


def test_submodules_exist():
    """Upstream submodule directories exist."""
    for path in EXPECTED_SUBMODULES:
        submodule_dir = REPO_ROOT / path
        assert submodule_dir.is_dir(), (
            f"Submodule not checked out: {path}. " f"Run: git submodule update --init"
        )


def test_submodules_pinned():
    """Upstream submodules are at the expected pinned revisions."""
    for path, expected_sha in EXPECTED_SUBMODULES.items():
        submodule_dir = REPO_ROOT / path
        if not submodule_dir.is_dir():
            continue
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=submodule_dir,
            capture_output=True,
            text=True,
        )
        actual_sha = result.stdout.strip()
        assert actual_sha == expected_sha, (
            f"{path} at {actual_sha[:12]}, expected {expected_sha[:12]}. "
            f"Run: git submodule update --init"
        )
