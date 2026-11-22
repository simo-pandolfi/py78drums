#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
mix.py — re-mix the stems of a sequencer render with other GAIN values (P4, V1)
================================================================================
wdf_rithm.py saves, next to every WAV, the raw output of each voice in
out/stems/<name>.npz. The master bus is linear in the voices (mix() of
wdf_rithm.py: stem * GAIN * bus load gain, summed, normalized to a peak of
0.5), so a new balance needs no new render: with the GAIN of wdf_rithm.py
the WAV is the one of the render, sample for sample.

GAIN changes only the level of each voice. The balance inside the noise
section (snare against cymbals and maracas: VR9, B1) is part of the circuit
and needs a new render.

Usage (from the py78drums root or from sequencer/):
    python sequencer/mix.py out/stems/pe78_vintage_rock_100bpm_2bar.npz noise=3.5 bd=0.35
    python sequencer/mix.py <stems.npz> [voice=gain ...] [-o out.wav]

voice: bd, cl, lb, cd, hb (tonal) or noise (snare + cymbals + maracas);
voices not given keep the GAIN of wdf_rithm.py. Default output:
out/mix/<name>_<gains>.wav. Printed: the GAIN used (ready to paste into
wdf_rithm.py), the peak of the bus and the RMS of each voice on the bus.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wdf_rithm as W  # noqa: E402

MIX_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'mix'))


def parse(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    path, gains, out = argv[0], {}, None
    rest = argv[1:]
    while rest:
        a = rest.pop(0)
        if a == "-o":
            if not rest:
                sys.exit("error: -o needs a file name")
            out = rest.pop(0)
        elif "=" in a:
            k, v = a.split("=", 1)
            if k not in W.VOICES:
                sys.exit(f"error: unknown voice '{k}' (voices: {', '.join(W.VOICES)})")
            gains[k] = float(v)
        else:
            sys.exit(f"error: unexpected argument '{a}'")
    return path, gains, out


def main(argv):
    path, changed, out = parse(argv)
    stems, meta = W.load_stems(path)
    gain = dict(W.GAIN)
    gain.update(changed)
    master, peak = W.mix(stems, gain, meta["g_tonal"], meta["g_noise"])

    print(f"{meta['rhythm']}  {meta['bpm']} BPM  {meta['bars']} bars  ({os.path.basename(path)})")
    print("GAIN = {" + ", ".join(f"'{k}': {gain[k]:g}" for k in W.VOICES if k in gain) + "}")
    print(f"bus peak before normalization: {peak:.4g}")
    tot = np.sqrt(np.mean((master * peak / 0.5) ** 2)) if peak > 1e-9 else 0.0
    for k in W.VOICES:
        if k in stems:
            g = meta["g_noise"] if k == "noise" else meta["g_tonal"]
            r = np.sqrt(np.mean((stems[k] * gain[k] * g) ** 2))
            rel = 20 * np.log10(r / tot) if r > 0 and tot > 0 else float("-inf")
            mark = "  *" if k in changed else ""
            print(f"  {k:6s} gain {gain[k]:<8g} RMS on the bus {rel:+6.1f} dB of the mix{mark}")

    if out is None:
        tag = "_".join(f"{k}{gain[k]:g}" for k in W.VOICES if k in changed) or "gain"
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(MIX_DIR, f"{name}_{tag}.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    W.save_wav(master, out, fs=int(meta["fs"]))


if __name__ == "__main__":
    main(sys.argv[1:])
