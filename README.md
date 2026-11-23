# py78drums

A Python model of the **DIY drum machine circuit** published in *Practical
Electronics*, January 1978 (PE78), modelled with Wave Digital Filters (WDF),
with rhythm patterns inspired by the *Nuova Elettronica* n. 60–61 build
(June–July 1978).

This is a **research project in progress**, developed as a study in preparation
for a Rust port as an audio plugin. The models are validated against LTspice
simulations; they have not been verified against a physical PE78 unit.

---

## Motivation

This project has a personal origin. In the early 1980s I built a DIY electronic
drum machine following the circuit published in *Nuova Elettronica* n. 60–61
(June–July 1978). The unit was eventually lost, and decades later I chose it as
a study subject for virtual analog synthesis with Wave Digital Filters — partly
for the technical interest, partly for the personal connection.

When researching the circuit online I first found the *Practical Electronics*
version (January 1978), which I remembered as very similar, and started the
implementation from that schematic. I later found the original *Nuova Elettronica*
circuit, which differs in one key aspect: it uses the **SGS-Ates M252AA** rhythm
generator instead of the M253 used in PE78, and includes one additional instrument
voice. Several near-identical designs can be found in other publications of the
period.

The implementation therefore combines the PE78 circuit (used as the primary
schematic reference, component names and LTspice validation target) with
M252AA-inspired rhythm patterns, which correspond to the sequencer in the
original *Nuova Elettronica* build.

---

## Background

The model has nine percussion voices, built with two synthesis approaches:

- **Twin-T oscillators** (Bass Drum, Hi Bongo, Low Bongo, Claves, Conga) —
  damped oscillators around a CD4011A gate, triggered by edge-detection
  circuits (Fig. 7)
- **Filtered noise** (Snare Drum, Long Cymbal, Short Cymbal, Maracas) —
  a shared transistor noise source, RC envelope networks and two BJT
  common-emitter stages with inductive collector loads (Fig. 6)

The WDF methodology models each subcircuit at the component level, preserving
the physical behaviour of reactive elements (capacitors, inductors). The
nonlinear elements (diodes, BJT base-emitter junctions, the CD4011A gate) are
grouped at R-type roots and solved by Newton-Raphson at every sample.

The Conga is not in PE78: it is taken from SGS-Ates Technical Note 131,
with the same edge detector and twin-T as the other tonal voices and
different capacitors. In the circuit it would need a fifth gate, since the
CD4011A of the tonal section has four.

The four noise voices share the noise node, the balance pot VR9 and the
output network, so they are modelled as one circuit (`NoiseVoices`). The
default position of VR9 is 80/220 (VR9U 80 kΩ), chosen for the level balance
of the mix; the LTspice schematic has 25/220. The
tonal and noise sections drive the same output bus; their mutual load is
applied as fixed mid-band gains (`bus_load_gains()`).

The sequencer (`wdf_rithm.py`) adopts the 32-step, two-bar architecture
inspired by the SGS-Ates M252AA chip, allowing for rhythmic variations
between the first and second bar, while featuring a custom set of original patterns.

---

## Project structure

```
py78drums/
├── ltspice/             # LTspice schematics of Fig. 6 (.asc, netlist)
├── pe78/
│   ├── drums.py         # TonalDrum, voice parameters, bus_load_gains(); re-exports NoiseVoices
│   ├── twint.py         # Twin-T voices: edge detector, twin-T, CD4011A gate (Fig. 7)
│   ├── noise_voices.py  # Snare, Long/Short Cymbal, Maracas as one circuit (Fig. 6)
│   ├── noise.py         # Noise source TR2 / B1: single definition of the level
│   ├── m253.py          # M253 trigger outputs as seen by the voices
│   └── wdf_nlroot.py    # R-type root with nonlinear ports, solved by Newton
├── sequencer/
│   ├── wdf_rithm.py     # Rhythm sequencer: renders a pattern to stems (.npz)
│   ├── mix.py           # Mixes the stems into a WAV with the voice gains
│   ├── patterns.py      # Original rhythms inspired by the 70s, 80s and later
│   ├── full_patterns.bat # Renders all the patterns (Windows)
│   └── pe78_*.wav       # Example rhythms
├── tools/               # LTspice comparison, reference solutions, sample-rate checks
├── tests/               # Tests (pytest)
├── bench/               # Single-voice test scripts and their outputs
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
scipy
soundfile
pywdf @ git+https://github.com/simo-pandolfi/pywdf-ref.git@7cdf940564e70ab1ed2eb18c04985fb266721306
```

The models use [pywdf-ref](https://github.com/simo-pandolfi/pywdf-ref), a copy of
pywdf with three corrections (see [Dependencies](#dependencies)), pinned to a
commit.

To use a local pywdf checkout instead of the installed package, set the
environment variable `PYWDF_ROOT` to its folder. Leave it unset to use the
pinned pywdf-ref.

---

## Usage

### Render a rhythm pattern

`wdf_rithm.py` is interactive. You can run it and follow the on-screen menu to
choose a rhythm, tempo, and duration:

```bash
python wdf_rithm.py
```

You can also pass parameters directly as command-line arguments:
```
python wdf_rithm.py <rhythm_number> [bpm] [bars]
```

`bpm` defaults to 100; `bars` is the number of pattern cycles (32 steps
each), default 2.

Example:

```
# Render 4 cycles of the 3rd rhythm at 128 BPM
python wdf_rithm.py 3 128 4
```

The sequencer computes every voice sample by sample at 96 kHz, with trigger
pulses of 20 ms. It saves the mix as a normalized 16-bit WAV in the current
folder (e.g. `pe78_vintage_rock_100bpm_2bar.wav`) and the raw output of each
voice (stems) as a `.npz` file in `out/stems/`.

> **Technical Note on Triggering**: In the original PE78 circuit, the snare
> output of the M253 also drives the Hi Bongo (HB) trigger line through D6:
> every snare hit also triggers the Hi Bongo, while a Hi Bongo hit alone does
> not trigger the snare. The model includes D6 and the coupling between the
> snare front end and the HB line.

### Re-mix the stems

The master bus is linear in the voices, so a new balance needs no new render.
`mix.py` mixes the stems again with other gains:

```
python sequencer/mix.py <stems.npz> [voice=gain ...] [-o out.wav]

# Example
python sequencer/mix.py out/stems/pe78_vintage_rock_100bpm_2bar.npz noise=3.5 bd=0.35
```

Voices: `bd`, `cl`, `lb`, `cd`, `hb` (tonal) and `noise` (snare, cymbals and
maracas together). Voices not given keep the gains of `wdf_rithm.py`; with no
gains the WAV is the one of the render, sample for sample. The default output
is `out/mix/<name>_<gains>.wav`. The script prints the gains used, ready to
paste into `wdf_rithm.py`, and the level of each voice in the mix. The
balance inside the noise section (snare against cymbals and maracas) is part
of the circuit (VR9) and needs a new render.

### Example Audio

If you want to listen to the models without running the code, pre-generated
rhythms (2 cycles, 96 kHz, default gains) are available in the `sequencer/`
folder.

### Use individual voices

```python
from pe78.drums import TonalDrum, NoiseVoices, BDO_PARAMS, HBONGO_PARAMS, bus_load_gains

FS = 96000

nv = NoiseVoices(FS)                          # snare, cymbals, maracas: one circuit (vr9_pos=...)
bd = TonalDrum(FS, **BDO_PARAMS)              # Bass Drum
hb = TonalDrum(FS, **HBONGO_PARAMS, d6=True)  # Hi Bongo, also driven by the snare
g_noise, g_tonal = bus_load_gains()           # mutual load on the output bus

def tick(v_sd=0.0, v_lc=0.0, v_sc=0.0, v_mr=0.0, v_bd=0.0, v_hb=0.0):
    # Noise section first: HB gets the snare line of the same sample,
    # the snare gets the D6 current of HB of the previous sample.
    y_noise = nv.tick(v_sd, v_lc, v_sc, v_mr, hb_i_d6=hb.voice.i_d6)
    y_tonal = bd.tick(v_bd) + hb.tick(v_hb, line=nv.hb_line())
    return g_noise * y_noise + g_tonal * y_tonal

sample = tick(v_sd=4.85, v_bd=4.85)
```

The trigger inputs are gates (above 0.5 V: on), modelled on the M253 outputs:
4.85 V behind 350 Ω when on, open circuit when off (`pe78/m253.py`). Each
`tick()` call advances the model by one sample and returns the voltage on the
pre-amp input, in volts. The tonal voices add linearly on their output
network; the other tonal voices are built in the same way from
`CONGA_PARAMS`, `LBONGO_PARAMS` and `CLAVES_PARAMS`.

---

## Validation

The noise voices (Fig. 6) are compared against the LTspice schematic in
`ltspice/` with `tools/ltspice_compare.py` (gates read from the `.asc`, 17 ms).
`tools/*_ref.py` are independent reference solutions of single subcircuits,
`tools/rate_compare.py` compares the voices at 48, 96 and 192 kHz, and
`tests/` holds the tests of each subcircuit and of the confirmed bugs.

---

## Known limitations

- **Output levels** are calibrated against the LTspice simulation, not a
  physical PE78 unit.
- **Noise source**: TR2 (BC108B, base-emitter junction in avalanche) is
  modelled as the LTspice behavioural current source B1, not as a physical
  noise measurement.
- **CD4011A gate**: one square-law model per transistor stack, fitted at the
  switching point; at full drive it gives about 6 mA against 1.1–1.2 mA of
  the typical curves.
- **TR3/TR4** are linear in the collector (no saturation, no Early effect).
- **One-sample couplings**: a few weak couplings between subcircuits use the
  value of the previous sample (TR3 base → snare front end, maracas ↔ ENV
  node, D6 current of the Hi Bongo → snare front end).
- **Output bus**: the mutual load of the tonal and noise sections is a fixed
  mid-band gain, within 0.06 dB from 200 Hz to 5 kHz and up to 0.6 dB at
  50 Hz against the full AC solution.

---

## References

- *Practical Electronics*, January 1978, pp. 24–25 — PE78 circuit schematic
- *Nuova Elettronica*, n. 60–61, June–July 1978 — original build reference
- A. Battaiotto, G. Ronzi, "M252 & M253 rhythm generators for electronic
  organs", SGS-Ates Technical Note 131, March 1979 — source of the Conga voice
- SGS-Ates — *MOS and special CMOS/MOS*, 1st edition, Nov. 1979 — M252/M253
  and other rhythm IC tables
- RCA — CD4011A data, File No. 479 — gate characteristics
- Fettweis, A. (1986). Wave Digital Filters: Theory and Practice.
  *Proceedings of the IEEE*, 74(2)
- Werner, K. J., Nangia, V., Smith, J. O., & Abel, J. S. (2015). Resolving
  Wave Digital Filters with Multiple/Multiport Nonlinearities. *DAFx-15*
- Olsen, M. J., Werner, K. J., & Smith, J. O. (2016). Resolving Grouped
  Nonlinearities in Wave Digital Filters using Iterative Techniques. *DAFx-16*
- Werner, K. J., Olsen, M. J., Rest, M., & Parker, J. D. (2017). Generalizing
  Root Variable Choice in Wave Digital Filters with Grouped Nonlinearities.
  *DAFx-17*
- Werner, K. J., Bernardini, A., Smith, J. O., & Sarti, A. (2018). Modeling
  Circuits with Arbitrary Topologies and Active Linear Multiports Using Wave
  Digital Filters. *IEEE Transactions on Circuits and Systems I*, 65(12),
  4233–4246. https://doi.org/10.1109/TCSI.2018.2837912

---

## Dependencies

- [pywdf-ref](https://github.com/simo-pandolfi/pywdf-ref) — copy of
  [pywdf](https://github.com/gusanthon/pywdf) at commit `3a1dac9` with three
  corrections, one commit each (branch `fixes`, pinned at `7cdf940`):
  - E1: `RootRTypeAdaptor` evaluation order
  - E2: `RTypeAdaptor` up-port reflected wave
  - E3: `OpenCircuit` port resistance
- [pywdf](https://github.com/gusanthon/pywdf) — WDF framework (MIT)
  Anthon, G., Lizarraga-Seijas, X., Font, F. (2023). *PYWDF: an open source
  library for prototyping and simulating wave filter circuits in Python*. DAFx23.
- [chowdsp_wdf](https://github.com/Chowdhury-DSP/chowdsp_wdf) — C++ WDF library
  on which pywdf is based (Jatin Chowdhury)
- numpy, scipy, soundfile

---

## AI assistance

This project was developed with extensive use of **Claude (Anthropic)** as an
AI assistant, contributing to WDF circuit decomposition, scattering matrix
derivation, debugging of sign conventions and impedance mismatches, level
calibration, and validation methodology.

---

## License

Copyright © 2026 Simone Pandolfi.

This project is dual-licensed under the [MIT License](LICENSE-MIT) and the [Apache License, Version 2.0](LICENSE-APACHE).
You may choose which license you'd like to use.
