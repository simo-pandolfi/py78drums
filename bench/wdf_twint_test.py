"""
PE 1978 Virtual Analog DIY e-Drums
Copyright (c) 2026 Simone Pandolfi
SPDX-License-Identifier: MIT OR Apache-2.0
==============================================================
WDF Twin-T Drum Voices — Analysis and VR4 range finder
Tests and validates all five tonal drum voices.

Usage (from bench root):
    python -m pe78.wdf_twint_test
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pe78.twint import EdgeDetector, TwinTDrum
from pe78.drums import (TonalDrum,
    BDO_PARAMS, CONGA_PARAMS, HBONGO_PARAMS, LBONGO_PARAMS, CLAVES_PARAMS)

FS = 48000
HIT_SAMPLES        = 480      # trigger pulse duration: 10 ms at 48 kHz
TOTAL_SAMPLES      = 33600    # analysis window: ~0.7 s per voice
RFB_TOTAL_SAMPLES  = 48000    # vr4 scan window: 1 s per voice
SILENCE_THRESHOLD  = 1e-4     # RMS above this -> self-oscillation detected
SILENCE_DURATION   = 0.5      # seconds of post-decay silence to measure


# =============================================================================
#  Utilities
# =============================================================================

def simulate(drum, fs=FS, total=TOTAL_SAMPLES, hit=HIT_SAMPLES):
    """Generate a single hit and return the DC-removed, normalised signal."""
    out = np.array([drum.tick(5.0 if i < hit else 0.0)
                    for i in range(total)])
    out -= out.mean()
    peak = np.max(np.abs(out))
    if peak > 0:
        out /= peak * 2     # normalise to ±0.5
    return out


def get_peak_freq(signal, fs=FS):
    """Return frequency of the spectral peak in Hz."""
    mag  = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(len(signal), 1.0 / fs)
    return freq[np.argmax(mag)]


def get_t60(signal, fs=FS):
    """Return time in ms for signal to decay to -60 dB from peak."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return 0.0
    db  = 20.0 * np.log10(np.abs(signal) / peak + 1e-12)
    idx = np.where(db > -60)[0]
    return (idx[-1] / fs * 1000.0) if len(idx) > 0 else 0.0


def get_t40(signal, fs=FS):
    """Return time in ms for signal to decay to -40 dB from peak."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return 0.0
    db  = 20.0 * np.log10(np.abs(signal) / peak + 1e-12)
    idx = np.where(db > -40)[0]
    return (idx[-1] / fs * 1000.0) if len(idx) > 0 else 0.0


def measure_residual_rms(drum, fs=FS,
                         hit=HIT_SAMPLES,
                         decay_samples=None,
                         silence_dur=SILENCE_DURATION):
    """
    Measure RMS of the residual signal after the sound has had time to decay.

    Method:
      1. Fire one trigger hit.
      2. Let the drum run for decay_samples (default: 2x T60 budget = 2 s).
      3. Measure RMS of the next silence_dur seconds without any trigger.

    A high residual RMS indicates self-oscillation.

    Parameters
    ----------
    drum : TonalDrum
        Fresh drum instance (state must be at rest).
    fs : int
    hit : int
        Trigger pulse length in samples.
    decay_samples : int or None
        Samples to run after trigger before measuring residual.
    silence_dur : float
        Duration in seconds of the residual measurement window.

    Returns
    -------
    float
        RMS of residual signal [V].
    """
    if decay_samples is None:
        decay_samples = RFB_TOTAL_SAMPLES

    # Fire trigger
    for i in range(decay_samples):
        drum.tick(5.0 if i < hit else 0.0)

    # Measure residual
    n_silence = int(fs * silence_dur)
    residual  = np.array([drum.tick(0.0) for _ in range(n_silence)])
    return float(np.sqrt(np.mean(residual ** 2)))


# =============================================================================
#  VR4 range finder
# =============================================================================

def find_vr4_range(params, fs=FS,
                   r_min=0, r_max=470e3, n_steps=20,
                   silence_thr=SILENCE_THRESHOLD,
                   t60_min_ms=30.0):
    """
    Scan VR4 values and classify each as:
      BLOCKED    : T60 < t60_min_ms  (insufficient oscillation)
      USEFUL     : no self-oscillation and T60 >= t60_min_ms
      SELF-OSC.  : residual RMS after decay > silence_thr

    For each VR4 value a fresh drum instance is created (clean state).
    Numerical overflow (e.g. Claves at extreme VR4) is caught and the
    point is marked as unstable (T60=0, RMS=inf).

    Parameters
    ----------
    params : dict
        Instrument parameter dictionary (BDO_PARAMS etc.).
    fs : int
    r_min, r_max : float
        VR4 scan range in Ohm.
    n_steps : int
        Number of scan points.
    silence_thr : float
        RMS threshold for self-oscillation detection.
    t60_min_ms : float
        Minimum acceptable T60 in ms.

    Returns
    -------
    r_vals, t60_vals, rms_vals : np.ndarray
    r_useful_min, r_useful_max : float (NaN if no useful range found)
    """
    r_vals   = np.linspace(r_min, r_max, n_steps)
    t60_vals = np.zeros(n_steps)
    rms_vals = np.zeros(n_steps)

    for k, r in enumerate(r_vals):
        drum = TonalDrum(fs, **params)
        drum.set_decay(r)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                # Measure post-decay residual (correct self-oscillation test)
                rms = measure_residual_rms(drum, fs)
                # Regenerate signal for T60 (fresh instance, same VR4)
                drum2 = TonalDrum(fs, **params)
                drum2.set_decay(r)
                sig  = simulate(drum2, fs, RFB_TOTAL_SAMPLES)
                t60  = get_t60(sig, fs)
        except (RuntimeWarning, FloatingPointError, OverflowError):
            # Numerical instability at this VR4 value -> mark as unstable
            t60 = 0.0
            rms = float('inf')

        t60_vals[k] = t60
        rms_vals[k] = rms

    # Useful range: sufficient T60 AND no self-oscillation
    useful_mask = (t60_vals >= t60_min_ms) & np.isfinite(rms_vals) & (rms_vals < silence_thr)
    if useful_mask.any():
        r_useful_min = r_vals[useful_mask][0]
        r_useful_max = r_vals[useful_mask][-1]
    else:
        r_useful_min = r_useful_max = float('nan')

    return r_vals, t60_vals, rms_vals, r_useful_min, r_useful_max


# =============================================================================
#  Instrument analysis and plots
# =============================================================================

INSTRUMENTS = [
    ("Bass Drum",  BDO_PARAMS,    "bass_drum.wav",   "tab:blue"),
    ("Conga",      CONGA_PARAMS,  "conga_drum.wav",  "tab:orange"),
    ("Hi Bongo",   HBONGO_PARAMS, "hbongo_drum.wav", "tab:green"),
    ("Low Bongo",  LBONGO_PARAMS, "lbongo_drum.wav", "tab:red"),
    ("Claves",     CLAVES_PARAMS, "claves_drum.wav", "tab:purple"),
]

# --- Compute and print summary table ---
print("=" * 60)
print(f"{'Voice':<12} {'Freq (Hz)':>10} {'T40 (ms)':>10} {'T60 (ms)':>10}")
print("=" * 60)

signals = {}
for name, params, wav_name, color in INSTRUMENTS:
    drum = TonalDrum(FS, **params)
    sig  = simulate(drum)
    freq = get_peak_freq(sig)
    t40  = get_t40(sig)
    t60  = get_t60(sig)
    wavfile.write(wav_name, FS, sig.astype(np.float32))
    signals[name] = (sig, color, freq, t40, t60)
    print(f"{name:<12} {freq:>10.1f} {t40:>10.1f} {t60:>10.1f}")

print("=" * 60)


# --- Plot 1: waveform + spectrum, 5 voices ---
fig, axes = plt.subplots(5, 2, figsize=(14, 20),
                         constrained_layout=True)
fig.suptitle("WDF Twin-T Drum Voices — Waveform and Spectrum", fontsize=13)

t_ms = np.arange(TOTAL_SAMPLES) / FS * 1000.0

for row, (name, _, _, _) in enumerate(INSTRUMENTS):
    sig, col, freq, t40, t60 = signals[name]

    # Waveform
    ax_t = axes[row, 0]
    ax_t.plot(t_ms, sig, color=col, linewidth=0.8)
    ax_t.set_title(f"{name}  {freq:.1f} Hz  T40={t40:.0f} ms  T60={t60:.0f} ms",
                   fontsize=8, pad=3)
    ax_t.set_xlabel("time (ms)", fontsize=8)
    ax_t.set_ylabel("amplitude", fontsize=8)
    ax_t.set_xlim(0, 300 if t60 < 250 else t_ms[-1])
    ax_t.tick_params(labelsize=7)
    ax_t.grid(True, alpha=0.4)

    # Spectrum
    ax_f = axes[row, 1]
    mag     = np.abs(np.fft.rfft(sig))
    freq_ax = np.fft.rfftfreq(len(sig), 1.0 / FS)
    ax_f.plot(freq_ax, mag, color=col, linewidth=0.8)
    ax_f.axvline(freq, color='k', linestyle='--', linewidth=0.8, alpha=0.6,
                 label=f"peak {freq:.1f} Hz")
    ax_f.set_xlim(0, min(max(freq * 3, 1500), 5000))
    ax_f.set_xlabel("frequency (Hz)", fontsize=8)
    ax_f.set_ylabel("magnitude", fontsize=8)
    ax_f.tick_params(labelsize=7)
    ax_f.legend(fontsize=7)
    ax_f.grid(True, alpha=0.4)

plt.savefig("drum_analysis.png", dpi=120, bbox_inches='tight')
plt.show()


# --- Plot 2: VR4 useful range for all voices ---
print("\n" + "=" * 60)
print("VR4 useful range scan (damped oscillation)")
print("=" * 60)

fig2, axes2 = plt.subplots(5, 1, figsize=(10, 20),
                            constrained_layout=True)
fig2.suptitle("VR4 useful range per voice", fontsize=12)

for row, (name, params, _, color) in enumerate(INSTRUMENTS):
    print(f"\n{name}...")
    t60_min = 20.0 if name == "Claves" else 40.0
    r_vals, t60_vals, rms_vals, r_min_u, r_max_u = find_vr4_range(
        params, FS,
        r_min=0, r_max=470e3, n_steps=20,
        t60_min_ms=t60_min
    )

    ax   = axes2[row]
    r_k  = r_vals / 1e3

    # T60 curve (left axis)
    finite_t60 = np.where(np.isfinite(t60_vals), t60_vals, np.nan)
    ax.plot(r_k, finite_t60, color=color, label="T60 (ms)", linewidth=1.5)
    ax.set_ylabel("T60 (ms)", color=color, fontsize=8)
    ax.tick_params(axis='y', labelcolor=color, labelsize=7)

    # Residual RMS curve (right axis)
    ax2r = ax.twinx()
    finite_rms = np.where(np.isfinite(rms_vals), rms_vals, np.nan)
    # Cap display at 10x threshold to keep the axis readable
    rms_display = np.clip(finite_rms, 0, SILENCE_THRESHOLD * 10) * 1e4
    ax2r.plot(r_k, rms_display, color='gray', linestyle='--',
              linewidth=1.0, label="residual RMS ×10⁴")
    ax2r.axhline(SILENCE_THRESHOLD * 1e4, color='gray',
                 linestyle=':', linewidth=0.8, alpha=0.7,
                 label=f"threshold ({SILENCE_THRESHOLD:.0e})")
    ax2r.set_ylabel("RMS ×10⁴", color='gray', fontsize=8)
    ax2r.tick_params(axis='y', labelcolor='gray', labelsize=7)

    # Mark unstable points
    unstable = ~np.isfinite(rms_vals)
    if unstable.any():
        ax.scatter(r_k[unstable], np.zeros(unstable.sum()),
                   color='red', marker='x', s=40, zorder=5,
                   label="numerical overflow")

    # Shade useful range
    if not np.isnan(r_min_u):
        ax.axvspan(r_min_u / 1e3, r_max_u / 1e3,
                   alpha=0.15, color=color,
                   label=f"useful {r_min_u/1e3:.0f}–{r_max_u/1e3:.0f} kΩ")
        print(f"  VR4 useful: {r_min_u/1e3:.0f} kΩ  ->  {r_max_u/1e3:.0f} kΩ")
        t60_in_range = t60_vals[(r_vals >= r_min_u) & (r_vals <= r_max_u)]
        t60_in_range = t60_in_range[np.isfinite(t60_in_range)]
        if len(t60_in_range):
            print(f"  T60 range:  {t60_in_range[0]:.0f} ms  ->  {t60_in_range[-1]:.0f} ms")
    else:
        print("  No useful range found — check parameters")

    ax.set_title(name, fontsize=9, pad=3)
    ax.set_xlabel("vr4 (kΩ)", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')

plt.savefig("rfb_range.png", dpi=120, bbox_inches='tight')
plt.show()