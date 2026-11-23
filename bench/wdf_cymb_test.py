"""
PE 1978 Virtual Analog DIY e-Drums
Copyright (c) 2026 Simone Pandolfi
SPDX-License-Identifier: MIT OR Apache-2.0
==============================================================
Test and simulation for the cymbals and maracas (PE 1978, Fig. 6 — bottom section).
Adapted for NoiseVoices (pe78/noise_voices.py, C15); was PE78_Cymbals_Maracas.

Interface of NoiseVoices used here (snare input kept at 0 V):
    drum.tick(v_sd, v_lc, v_sc, v_mr, noise) -> V_out on RLoad
      noise : sigma = 1, scaled internally by NOISE_AMP (pe78/noise.py)
    State after each tick: _V_env, _V_c23, _V_c24, _V_c26, _V_col4, _V_out
    Constants: V_ON, R47, C23, R44, C24, R46, C26, R54, L2, hFE, Is_b4, Vt,
    R52, C29
  gm(t) is reconstructed here: gm = hFE * Is_b4 * exp(V_env / Vt) / Vt
  (no clamp: C11).

Test structure:
  1. Three separate pulses — LC t=50ms, SC t=550ms, MR t=1050ms
  2. Printed analysis: theoretical time constants, peaks, durations
  3. 5-panel plot: V_env, V_c23, V_c24, V_c26, V_out
  4. Normalized WAV
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pe78.noise_voices import NoiseVoices

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
FS = 96000
T  = 2.0
N  = int(T * FS)
PW = 0.020           # pulse duration [s]

T_LC = 0.05
T_SC = 0.55
T_MR = 1.05

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(seed=42):
    """
    Runs the simulation with three separate pulses (LC, SC, MR).

    Note on noise:
      noise_sample has sigma=1; tick() scales it by NOISE_AMP =
      noise_amp(fs) (pe78/noise.py). Do not prescale the noise.
    """
    drum = NoiseVoices(FS)          # C15: snare idle

    rng   = np.random.default_rng(seed)
    noise = rng.standard_normal(N)   # sigma = 1

    V_out = np.zeros(N)
    V_env = np.zeros(N)
    V_c23 = np.zeros(N)
    V_c24 = np.zeros(N)
    V_c26 = np.zeros(N)

    # Trigger windows
    lc_on = slice(int(T_LC * FS), int((T_LC + PW) * FS))
    sc_on = slice(int(T_SC * FS), int((T_SC + PW) * FS))
    mr_on = slice(int(T_MR * FS), int((T_MR + PW) * FS))

    trig_lc = np.zeros(N)
    trig_sc = np.zeros(N)
    trig_mr = np.zeros(N)
    trig_lc[lc_on] = drum.V_ON
    trig_sc[sc_on] = drum.V_ON
    trig_mr[mr_on] = drum.V_ON

    for i in range(N):
        V_out[i] = drum.tick(
            0.0, trig_lc[i], trig_sc[i], trig_mr[i],
            noise[i]          # sigma ≈ 1; internal NOISE_AMP handles scaling
        )

        # State updated after tick()
        V_env[i] = drum._V_env   # ENV node (physical voltage)
        V_c23[i] = drum._V_c23   # LC hold capacitor
        V_c24[i] = drum._V_c24   # SC hold capacitor
        V_c26[i] = drum._V_c26   # MR hold capacitor

    # gm derived from V_env with the base law of the bases root (Shockley,
    # Is_b4, Vt) and Ic = hFE*Ib: gm = hFE · Is_b4 · exp(V_env / Vt) / Vt
    # (no clamp, C11).
    Ic_em = drum.hFE * drum.Is_b4 * np.exp(np.clip(V_env / drum.Vt, -40.0, 40.0))
    gm = Ic_em / drum.Vt

    return {
        'V_out': V_out,
        'V_env': V_env,
        'V_c23': V_c23,
        'V_c24': V_c24,
        'V_c26': V_c26,
        'gm':    gm,
        'drum':  drum,
    }


# ---------------------------------------------------------------------------
# Textual analysis
# ---------------------------------------------------------------------------
def print_analysis(res):
    drum = res['drum']

    print('=== Time constants (theoretical) ===')
    print(f'  LC  τ = R47×C23  = {drum.R47 * drum.C23 * 1000:.0f} ms')
    print(f'  SC  τ = R44×C24  = {drum.R44 * drum.C24 * 1000:.0f} ms')
    print(f'  MR  τ = R46×C26  = {drum.R46 * drum.C26 * 1000:.0f} ms')
    print(f'  (L2+Rser)||R54 pole at = {(drum.R54 + drum.L_RSER) / (2 * np.pi * drum.L2):.0f} Hz')

    # C29 pole with r_pi from the base law at V_env = 0.7 V (typical point)
    V_typ    = 0.7
    Ic_typ   = drum.hFE * drum.Is_b4 * np.exp(np.clip(V_typ / drum.Vt, -40.0, 40.0))
    gm_typ   = Ic_typ / drum.Vt
    r_pi_typ = drum.hFE / gm_typ
    print(f'  C29 pole (r_pi @ 0.7 V)  = {1 / (2 * np.pi * r_pi_typ * drum.C29):.0f} Hz')
    print()

    gm = res['gm']
    gm_ss_lc = drum.hFE * drum.Is_b4 * np.exp(np.clip(0.62 / drum.Vt, -40, 40)) / drum.Vt
    print('=== TR4 (base law, Ic = hFE*Ib) ===')
    print(f'  Is_b          = {drum.Is_b4:.3e} A')
    print(f'  gm_ss @ 620mV = {gm_ss_lc*1e3:.2f} mA/V  (LC operating point)')
    print(f'  gm_max        = {gm.max()*1e3:.3f} mA/V  at t = {gm.argmax()/FS*1000:.1f} ms')
    # "active" = gm > 1% of its global maximum
    gm_thresh = gm.max() * 0.01
    print(f'  gm > 1% gm_max for {(gm > gm_thresh).sum() / FS * 1000:.0f} ms total')
    print()

    for name, sig, label in [
        ('V_env', res['V_env'], 'V_env'),
        ('V_c23', res['V_c23'], 'V_c23 (LC)'),
        ('V_c24', res['V_c24'], 'V_c24 (SC)'),
        ('V_c26', res['V_c26'], 'V_c26 (MR)'),
        ('V_out', res['V_out'], 'V_out'),
    ]:
        pk = np.max(np.abs(sig))
        active = np.where(np.abs(sig) > pk * 0.02)[0]
        dur = (active[-1] - active[0]) / FS * 1000 if len(active) > 1 else 0
        print(f'  {label:12s}  peak={pk:.5f} V   active≈{dur:.0f} ms')


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(res, outpath):
    t_ms = np.arange(N) / FS * 1000

    trig_info = [
        (T_LC, T_LC + PW, 'LC', '#e74c3c'),
        (T_SC, T_SC + PW, 'SC', '#f39c12'),
        (T_MR, T_MR + PW, 'MR', '#2ecc71'),
    ]

    def add_trig_lines(ax):
        ylim = ax.get_ylim()
        for t0, t1, label, col in trig_info:
            ax.axvline(t0 * 1000, color=col, ls='--', lw=1.2, alpha=0.8)
            ax.axvline(t1 * 1000, color=col, ls=':',  lw=0.9, alpha=0.5)
            ax.text(t0 * 1000 + 8, ylim[0] + (ylim[1] - ylim[0]) * 0.85,
                    label, color=col, fontsize=7, fontweight='bold')

    panels = [
        (res['V_env'], 'V_env [V]', 'ENVELOPE (ENV node → base TR4)', '#3498db'),
        (res['V_c23'], 'V [V]',     'C23 — Long Cymbal (hold)',        '#e74c3c'),
        (res['V_c24'], 'V [V]',     'C24 — Short Cymbal (hold)',       '#f39c12'),
        (res['V_c26'], 'V [V]',     'C26 — Maracas (hold)',            '#2ecc71'),
        (res['V_out'], 'V [V]',     'Output on RLoad (VR9 wiper → R55 → C31)',           '#e67e22'),
    ]

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(14, 13), sharex=True)
    fig.subplots_adjust(hspace=0.45)

    for ax, (data, ylabel, title, color) in zip(axes, panels):
        ax.plot(t_ms, data, color=color, lw=0.9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim([0, T * 1000])
        ax.grid(True, alpha=0.3)
        add_trig_lines(ax)

    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle(
        'PE Noise Drums — NoiseVoices (cymbals)\n'
        'LC (t=50ms)  ·  SC (t=550ms)  ·  MR (t=1050ms)\n'
        'Bases root (TR3, TR4, N) + maracas root (D8, D9)',
        fontsize=11, fontweight='bold')

    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Plot saved: {outpath}')


# ---------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------
def save_wav(data, path, fs=FS):
    v = data / (np.max(np.abs(data)) + 1e-9) * 0.8
    wavfile.write(path, fs, (v * 32767).astype(np.int16))
    print(f'WAV saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Simulation NoiseVoices (cymbals, maracas) — three separate pulses...')
    res = simulate()
    print('Completed.\n')

    lc_end = int((T_LC + PW) * FS)
    V_env_ss = np.mean(res['V_env'][lc_end - int(0.005*FS) : lc_end])
    print(f"V_ENV_ss (end of LC trigger) = {V_env_ss*1000:.1f} mV  (target ~620)")

    print_analysis(res)


    plot(res, 'wdf_cymb_test.png')
    save_wav(res['V_out'], 'wdf_cymb_test.wav')