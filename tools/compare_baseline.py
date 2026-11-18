#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Compare two runs of tools/baseline.py (for example original vs corrected pywdf).

Usage (from the py78drums root):
    python tools/compare_baseline.py <A.npz> <B.npz> [--all]

For every signal: max |A-B|, RMS of A-B, max |A-B| relative to the peak of A.
For the outputs ("out"): peak, RMS, last sample above 1% of peak, dominant
frequency, A vs B. By default only outputs are listed; --all adds the internal
states (envelope voltages, collector, ...).
"""

import argparse
import json
import os

import numpy as np


def load(path):
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
        return np.load(path), json.load(f)


def fmt(v, spec=".3g"):
    return "-" if v is None else format(v, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--all", action="store_true", help="also list internal states")
    a = p.parse_args()
    sa, ra = load(a.a)
    sb, rb = load(a.b)
    print(f"A: {ra['pywdf']}\nB: {rb['pywdf']}")
    for k in ("seed", "noise_std"):
        if ra[k] != rb[k]:
            print(f"warning: {k} differs ({ra[k]} vs {rb[k]})")

    print(f"\n{'signal':40s} {'max|A-B|':>10s} {'rms(A-B)':>10s} {'rel':>9s}")
    for key in sorted(set(sa.files) & set(sb.files)):
        if not a.all and not key.endswith("/out"):
            continue
        x, y = sa[key], sb[key]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            print(f"{key:40s} non-finite values")
            continue
        d = x - y
        pk = np.max(np.abs(x))
        rel = np.max(np.abs(d)) / pk if pk > 0 else np.nan
        print(f"{key:40s} {np.max(np.abs(d)):10.3e} {np.sqrt(np.mean(d ** 2)):10.3e} {rel:9.2e}")

    print(f"\n{'case':28s} {'peak A':>10s} {'peak B':>10s} {'rms A':>10s} {'rms B':>10s}"
          f" {'last A ms':>9s} {'last B ms':>9s} {'f A Hz':>8s} {'f B Hz':>8s}")
    for case in sorted(set(ra["cases"]) & set(rb["cases"])):
        ma, mb = ra["cases"][case]["metrics"]["out"], rb["cases"][case]["metrics"]["out"]
        print(f"{case:28s} {fmt(ma.get('peak'), '10.3e')} {fmt(mb.get('peak'), '10.3e')}"
              f" {fmt(ma.get('rms'), '10.3e')} {fmt(mb.get('rms'), '10.3e')}"
              f" {fmt(ma.get('t_last_ms'), '9.1f')} {fmt(mb.get('t_last_ms'), '9.1f')}"
              f" {fmt(ma.get('f_dom_hz'), '8.1f')} {fmt(mb.get('f_dom_hz'), '8.1f')}")


if __name__ == "__main__":
    main()
