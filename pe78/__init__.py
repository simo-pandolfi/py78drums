#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
pe78: WDF model of the PE 1978 drum machine (Practical Electronics, January 1978).

Which pywdf is used (one rule for tests, bench, sequencer and tools): if the
environment variable PYWDF_ROOT is set, the folder it names (the one that
contains the `pywdf` package, for example a clone of pywdf-ref) is put first
on sys.path before any pe78 module imports pywdf; otherwise the installed
pywdf is used (requirements.txt: pywdf-ref at 7cdf940). The tools accept
--pywdf, which sets PYWDF_ROOT for their process.
"""

import os
import sys

_root = os.environ.get("PYWDF_ROOT")
if _root:
    _root = os.path.abspath(_root)
    if not os.path.isdir(os.path.join(_root, "pywdf")):
        raise ImportError(f"PYWDF_ROOT={_root}: no pywdf package in this folder")
    if _root in sys.path:
        sys.path.remove(_root)
    sys.path.insert(0, _root)
