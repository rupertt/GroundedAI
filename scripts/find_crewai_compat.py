"""
Find a CrewAI version that is compatible with this repo's pinned Chroma version.

We currently pin chromadb < 0.6 due to langchain-chroma. Newer CrewAI versions
depend on chromadb~=1.x, which conflicts.

This script downloads wheel metadata (no dependencies) for a list of candidate
versions and prints any Requires-Dist lines for chromadb.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def chromadb_requires_from_wheel(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as z:
        meta_files = [n for n in z.namelist() if n.endswith("METADATA") and ".dist-info/" in n]
        if not meta_files:
            return []
        meta = z.read(meta_files[0]).decode("utf-8", errors="replace")
    return [ln for ln in meta.splitlines() if ln.startswith("Requires-Dist: chromadb")]


def main() -> None:
    versions = sys.argv[1:] or [
        "0.70.1",
        "0.79.4",
        "0.86.0",
        "0.100.0",
        "0.120.0",
        "0.150.0",
        "0.175.0",
        "0.193.2",
        "0.201.0",
        "0.203.2",
        "1.0.0",
        "1.4.0",
        "1.7.2",
    ]

    pip = [sys.executable, "-m", "pip"]

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        for v in versions:
            print(f"--- crewai=={v}")
            try:
                run(pip + ["download", "--no-deps", f"crewai=={v}", "-d", str(out_dir)])
            except Exception as e:
                print(f"download_failed={type(e).__name__}")
                continue

            wheels = sorted(out_dir.glob(f"crewai-{v}-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not wheels:
                print("no_wheel_found")
                continue

            reqs = chromadb_requires_from_wheel(wheels[0])
            if not reqs:
                print("chromadb_requires=(none)")
            else:
                for r in reqs:
                    print(r)

            # Cleanup for the next iteration so we don't mix wheels.
            for p in out_dir.glob("*.whl"):
                p.unlink(missing_ok=True)

    # If pip created any cache artifacts in cwd, clean them (best effort).
    for name in ["pip-wheel-metadata", "build"]:
        p = Path(name)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()


