#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Test of the pywdf selection rule (pe78/__init__.py, P4).

Tests, bench, sequencer and tools take pywdf from the same place: the folder
named by PYWDF_ROOT if it is set, otherwise the installed pywdf. Before this
rule only the tests read PYWDF_ROOT; bench/ and sequencer/ failed with
"No module named 'pywdf'" without an installed pywdf.

1. With PYWDF_ROOT set (as for every test run here), a fresh interpreter that
   only has the repository on sys.path imports pe78.noise_voices and gets the
   pywdf of PYWDF_ROOT; skipped if PYWDF_ROOT is not set.
2. PYWDF_ROOT naming a folder without a pywdf package: ImportError with the
   folder in the message.

Usage (from the py78drums root):
    python -m pytest tests/test_pywdf_root.py
    python tests/test_pywdf_root.py
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = ("import sys; sys.path.insert(0, sys.argv[1]); "
        "import pe78.noise_voices, pywdf; print(pywdf.__file__)")


def _run(pywdf_root, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYWDF_ROOT"] = pywdf_root
    return subprocess.run([sys.executable, "-c", CODE, ROOT], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_pywdf_root_used():
    root = os.environ.get("PYWDF_ROOT")
    if not root:
        print("PYWDF_ROOT not set: skipped")
        return
    with tempfile.TemporaryDirectory() as cwd:
        r = _run(root, cwd)
    assert r.returncode == 0, r.stderr
    got = os.path.normcase(os.path.abspath(r.stdout.strip()))
    want = os.path.normcase(os.path.abspath(root))
    print(f"pywdf: {got}")
    assert got.startswith(want + os.sep), (got, want)


def test_pywdf_root_without_package():
    with tempfile.TemporaryDirectory() as empty:
        r = _run(empty, empty)
    assert r.returncode != 0
    assert "no pywdf package" in r.stderr and os.path.basename(empty) in r.stderr, r.stderr


if __name__ == "__main__":
    test_pywdf_root_used()
    test_pywdf_root_without_package()
    print("OK")
