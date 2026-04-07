"""
PE 1978 Virtual Analog DIY e-Drums
Copyright (c) 2026 Simone Pandolfi
SPDX-License-Identifier: MIT OR Apache-2.0
==============================================================
WDF Snare Drum Voice — Analysis
Test script for the Snare Drum model.
Includes diagnostics for the envelope generator circuit (D7 diode issue).
"""
import sys, os
print('\n'.join(sys.path))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

from pe78.drums import SnareDrum

# --- NORMALIZATION AND WAV EXPORT ---
def save_normalized_wav(filename, data, fs, target_db=-3):
    # Remove DC offset if present
    data = data - np.mean(data)

    # Peak normalization
    peak = np.max(np.abs(data))
    if peak > 0:
        # Calculate the factor to reach the target dB (e.g. -3dB = ~0.707)
        ratio = 10**(target_db / 20)
        normalized_data = (data / peak) * ratio
    else:
        normalized_data = data

    # Convert to Int16 for standard compatibility
    audio_int16 = (normalized_data * 32767).astype(np.int16)
    wavfile.write(filename, fs, audio_int16)
    print(f"File saved: {filename}")


# --- Simulation and Export ---
fs = 48000
snare = SnareDrum(fs)
duration = 0.5
n_samples = int(fs * duration)

# Signal generation
trigger = np.zeros(n_samples)
trigger[10:500] = 4.5 
noise = np.random.normal(0, 1, n_samples)
output = np.zeros(n_samples)

for i in range(n_samples):
#    output[i] = snare.model.process_sample(trigger[i], noise[i] * 0.04)
    output[i] = snare.tick(trigger[i])

peak   = np.max(np.abs(output))
active = np.sum(np.abs(output) > peak * 0.01) if peak > 1e-9 else 0
print(f"  Snare diagnostic: peak={peak:.5f} V, "
      f"active samples={active} ({active/fs*1000:.0f} ms)")
if peak < 1e-4:
    print("  → ALMOST ZERO SIGNAL: envelope bug in PE78_Snare "
          "(D7 not conducting). The snare will be silent until fixed.")
else:
    print("  → Snare OK")


save_normalized_wav("snare_noise.wav", output, fs)

plt.plot(output)
plt.title("WDF Snare Output (Collector Node)")
plt.show()