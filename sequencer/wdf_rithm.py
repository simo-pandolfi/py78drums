#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
wdf_rithm.py — PE 1978 Noise Drums - Rhythmic Sequencer
==============================================================
A 16-step sequencer featuring two-beat rhythms inspired by the M253 AC, 
utilizing all physical instrument models from the PE 1978 circuit.
The kit has been expanded with Conga Drums to support M252-style 
rhythmic synthesis.

Instruments:
  bd     Bass Drum          (TonalDrum)
  sd     Snare Drum         (NoiseVoices) — Latin/Rock switch
  hb     Hi Bongo           (TonalDrum) — shares the SD trigger
  lb     Low Bongo          (TonalDrum)
  cd     Conga Drum         (TonalDrum) — present on M252 AA
  lc/sc/mr Long/Short cymbal, Maracas (NoiseVoices, same object as sd)
  cl     Claves             (TonalDrum) — Latin/Rock switch

Notes on original PE 1978 wiring:
  In the PE 1978 circuit, the HB (Hi Bongo) bus is connected to 
  the SD (Snare Drum) output: every time the snare is triggered, 
  the HB receives the same pulse. The 'sd' line in the pattern 
  therefore activates BOTH instruments.
  
  OUT3 of the M253 drives Claves (Latin style) or Snare Drum 
  (Rock style) depending on the style of the selected rhythm.
"""

import sys, os
print('\n'.join(sys.path))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.io import wavfile
from patterns import PATTERNS

# pe78_fast is not used until it is realigned with pe78/ (it reverts C1).
from pe78.drums import (TonalDrum, NoiseVoices, bus_load_gains,
    BDO_PARAMS, CONGA_PARAMS, HBONGO_PARAMS, LBONGO_PARAMS, CLAVES_PARAMS)


# ---------------------------------------------------------------------------
# Session Parameters
# ---------------------------------------------------------------------------
FS           = 48000
BPM_DEFAULT  = 100
STEPS        = 32
HIT_SAMPLES  = 960       # Trigger pulse duration ≈ 20 ms @ 48 kHz

# Instrument gain on master bus (relative balancing).
# Tonal voices: 0.3046 = (R13/3.9M) / (1/5 * 10k/(78k + 10k)) keeps the mix
# unchanged after the twin-T output fix (pre-amp input, same node as sd/cymb).
GAIN = {
    'bd':    0.3046,
    'noise': 6.00,      # NoiseVoices (snare + cymbals + maracas, one output, C15)
    'hb':    0.3046,
    'lb':    0.3046,
    'cd':    0.3046,
    'cl':    0.3046,
}


# ---------------------------------------------------------------------------
# Noise voices
# ---------------------------------------------------------------------------
# Snare, long/short cymbal and maracas are one analog circuit (PE 1978 Fig. 6):
# shared noise node N, shared balance pot VR9 and output R55/C31. They are one
# persistent NoiseVoices object (C15) fed with the four triggers of the current
# step; its single output goes to the master bus with GAIN['noise'] (= the old
# snare gain: the snare level is unchanged, the cymbal level is now set by VR9).
# GAIN keeps the mix balance chosen by ear (C16, D2-B): with one gain for all
# voices the noise voices would be 25.9 dB lower than here (physical bus).
# bus_load_gains() applies the mutual load of the two sections on VR10.


def synthesize(rhythm_name, bpm=BPM_DEFAULT, num_bars=2,
               latin_mode=None, fs=FS):
    """
    Synthesizes the rhythm by calculating every single sample for all instruments,
    exactly as it would happen in a real-time audio engine in Rust.
    """
    if rhythm_name not in PATTERNS:
        raise ValueError(
            f"Rhythm '{rhythm_name}' not found.\n"
            f"Available: {list(PATTERNS.keys())}"
        )

    pattern = PATTERNS[rhythm_name]

    # Samples per 1/16th note step, 32 steps equal two bars
    step_samples  = int(60.0 / bpm / 4 * fs)   
    
    # Ternary time support: pattern can declare 'steps': 24
    # (or any other length) without needing -1 padding.
    pattern_steps = pattern.get('steps', STEPS)
    total_samples = step_samples * pattern_steps * num_bars
    master        = np.zeros(total_samples)


    # --- 1. Instrument Initialization (WDF Trees) ---
    print(f"\nInitializing instruments for '{rhythm_name}'...")
    inst_models = {}
    
    if 'bd' in pattern: inst_models['bd'] = TonalDrum(fs, **BDO_PARAMS)
    # The SD bus triggers Hi Bongo and Snare Drum simultaneously
    if 'hb' in pattern or 'sd' in pattern: inst_models['hb'] = TonalDrum(fs, **HBONGO_PARAMS)
    if 'lb' in pattern: inst_models['lb'] = TonalDrum(fs, **LBONGO_PARAMS)
    if 'cl' in pattern: inst_models['cl'] = TonalDrum(fs, **CLAVES_PARAMS)
    if 'cd' in pattern: inst_models['cd'] = TonalDrum(fs, **CONGA_PARAMS)

    # Snare, cymbals and maracas: one circuit (shared noise node, VR9, R55/C31)
    has_noise = any(k in pattern for k in ('sd', 'lc', 'sc', 'mr'))
    if has_noise:
        noise_model = NoiseVoices(fs)

    # Mutual load of the tonal and noise sections on the output bus (C16,
    # D1-B): both branches are always connected in the circuit.
    G_NOISE, G_TONAL = bus_load_gains(NoiseVoices.VR9_POS_DEFAULT)

    # --- 2. Sequence Extraction for convenience ---
    seq_bd = pattern.get('bd', [0]*STEPS)
    seq_cl = pattern.get('cl', [0]*STEPS)
    seq_sd = pattern.get('sd', [0]*STEPS)
    seq_hb = pattern.get('hb', [0]*STEPS)
    seq_lb = pattern.get('lb', [0]*STEPS)
    seq_cd = pattern.get('cd', [0]*STEPS)
    seq_lc = pattern.get('lc', [0]*STEPS)
    seq_sc = pattern.get('sc', [0]*STEPS)
    seq_mr = pattern.get('mr', [0]*STEPS)

    print(f"Tick-by-tick synthesis: {num_bars} bars @ {bpm} BPM...")

    # --- 3. Main Audio Loop (Sample-by-Sample) ---
    for i in range(total_samples):
        # Current timing calculation
        step_idx = (i // step_samples) % pattern_steps
        pos_in_step = i % step_samples
        
        # Trigger pulse generation (4.5V during the first HIT_SAMPLES)
        in_trig = pos_in_step < HIT_SAMPLES
        trig_v = 4.5 if in_trig else 0.0

        sample_mix = 0.0

        # Process discrete instruments
        if 'bd' in inst_models:
            v = trig_v if seq_bd[step_idx] == 1 else 0.0
            sample_mix += inst_models['bd'].tick(v) * GAIN['bd'] * G_TONAL

        if 'hb' in inst_models:
            # Sum of physical output from HB and SD nodes sharing the trigger
            v = trig_v if seq_hb[step_idx] or seq_sd[step_idx] == 1 else 0.0
            sample_mix += inst_models['hb'].tick(v) * GAIN['hb'] * G_TONAL

        if 'cl' in inst_models:
            v = trig_v if seq_cl[step_idx] == 1 else 0.0
            sample_mix += inst_models['cl'].tick(v) * GAIN['cl'] * G_TONAL


        if 'lb' in inst_models:
            v = trig_v if seq_lb[step_idx] == 1 else 0.0
            sample_mix += inst_models['lb'].tick(v) * GAIN['lb'] * G_TONAL

        if 'cd' in inst_models:
            v = trig_v if seq_cd[step_idx] == 1 else 0.0
            sample_mix += inst_models['cd'].tick(v) * GAIN['cd'] * G_TONAL

        # Noise voices: snare, cymbals, maracas (one circuit, one output)
        if has_noise:
            v_sd = trig_v if seq_sd[step_idx] == 1 else 0.0
            v_lc = trig_v if seq_lc[step_idx] == 1 else 0.0
            v_sc = trig_v if seq_sc[step_idx] == 1 else 0.0
            v_mr = trig_v if seq_mr[step_idx] == 1 else 0.0
            sample_mix += noise_model.tick(v_sd, v_lc, v_sc, v_mr) * GAIN['noise'] * G_NOISE

        # Write to master bus
        master[i] = sample_mix

    # --- 4. Final Normalization (-1 dBFS 0.891 or similar) ---
    peak = np.max(np.abs(master))
    if peak > 1e-9:
        master = (master / peak) * 0.5
    else:
        print("[WARNING] Silent master bus — check instruments.\n")

    print(f"[INFO] Master bus peak: {peak}.\n")

    return master


# ---------------------------------------------------------------------------
# WAV Saving
# ---------------------------------------------------------------------------
def save_wav(data, path, fs=FS):
    wavfile.write(path, fs, (data * 32000).astype(np.int16))
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Interactive Main
# ---------------------------------------------------------------------------
def main():
    rhythm_list = list(PATTERNS.keys())

    print("\n" + "="*54)
    print("  PE 1978 Noise Drums — M253 AC Sequencer (15 rhythms)")
    print("="*54)
    for i, name in enumerate(rhythm_list, 1):
        print(f"  {i:2d}.  {name:<14}")
    print("   0.  Exit")
    print("="*54)
    print("  Usage: <number> [bpm] [bars]   e.g.: 3 128 8")
    print()

    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])
    else:
        raw = input("Choice: ").strip()

    parts = raw.split()
    if not parts or parts[0] == '0':
        print("Exiting.")
        return

    try:
        idx = int(parts[0])
        if not (1 <= idx <= len(rhythm_list)):
            raise ValueError
    except ValueError:
        print(f"Invalid choice: '{parts[0]}'")
        return

    rhythm_name = rhythm_list[idx - 1]
    bpm         = int(parts[1]) if len(parts) > 1 else BPM_DEFAULT
    num_bars    = int(parts[2]) if len(parts) > 2 else 4

    print(f"\nRhythm: {rhythm_name}  |  {bpm} BPM  |  {num_bars} bars")

    audio = synthesize(rhythm_name, bpm=bpm, num_bars=num_bars)

    slug  = rhythm_name.lower().replace(' ', '_')
    fname = f"pe78_{slug}_{bpm}bpm_{num_bars}bar.wav"
    save_wav(audio, fname)


if __name__ == "__main__":
    main()