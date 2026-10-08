"""
DRAG waveform diagnostic script.

UNVERIFIED AI GENERATED CODE — PLEASE REVIEW CAREFULLY BEFORE RUNNING!

Plots:
  1. Raw qualang_tools output for X90 — establishes sign convention
  2. I and Q channels for X90, mX90, Y90, mY90 as built by config_helpers

Run directly:  python -m LabCode.Helpers.plot_drag_waveforms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from qualang_tools.config.waveform_tools import drag_cosine_pulse_waveforms

# ── Parameters — edit to match your qubit ────────────────────────────────────
AMP           = 0.3066
LENGTH_NS     = 20        # X90 length in ns
DRAG_BETA     = 1.0       # use 1.0 to see full DRAG shape
ANHARMONICITY = -155e6    # Hz, must be negative for transmon
DETUNING      = 0.0       # Hz
# ─────────────────────────────────────────────────────────────────────────────


def get_waveforms(amp, length, beta, anharm, detuning):
    raw = drag_cosine_pulse_waveforms(amp, length, beta, anharm, detuning=detuning)
    return np.array(raw[0]), np.array(raw[1])


I_raw, Q_raw = get_waveforms(AMP, LENGTH_NS, DRAG_BETA, ANHARMONICITY, DETUNING)

# ── Config-helpers sign convention (current) ─────────────────────────────────
# Forward gates: +raw[1]. Inverse gates: -raw[1].
# Channel assignments transcribed from config_helpers.py (COSINE branch). For each
# gate, (I, Q) are the physical I/Q output channels.
#   X90 :  I = X90_wf_      = +raw[0]   Q = X90_wf_der_  = +raw[1]
#   mX90:  I = Xm90_wf_     = -raw[0]   Q = Xm90_wf_der_ = -raw[1]
#   Y90 :  I = Y90_wf_der_  = -raw[1]   Q = Y90_wf_      = +raw[0]
#   mY90:  I = Ym90_wf_der_ = +raw[1]   Q = Ym90_wf_     = -raw[0]
# Note Y90 = i * X90 (a true +90 deg phase rotation), so its complex-envelope
# spectrum matches X90 (same DRAG asymmetry). Using +raw[1] for Y90.I instead would
# be i*conj(X90), a reflection that frequency-mirrors the spectrum (a plotting bug).
waveforms = {
    "X90":  {"I":  I_raw, "Q":  Q_raw},
    "mX90": {"I": -I_raw, "Q": -Q_raw},
    "Y90":  {"I": -Q_raw, "Q":  I_raw},
    "mY90": {"I":  Q_raw, "Q": -I_raw},
}

t = np.arange(LENGTH_NS)

# ── Figure 1: raw qualang_tools output ───────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig1.suptitle(
    f"Raw drag_cosine_pulse_waveforms output\n"
    f"amp={AMP}, length={LENGTH_NS} ns, β={DRAG_BETA}, η={ANHARMONICITY/1e6:.0f} MHz",
    fontsize=10,
)
axes[0].plot(t, I_raw, color="steelblue")
axes[0].set_title("raw[0]  →  I (cosine envelope)")
axes[0].set_xlabel("sample (ns)")
axes[0].axhline(0, color="k", lw=0.5)

axes[1].plot(t, Q_raw, color="tomato")
axes[1].set_title("raw[1]  →  Q (DRAG correction)")
axes[1].set_xlabel("sample (ns)")
axes[1].axhline(0, color="k", lw=0.5)

sign_rising = "POSITIVE" if Q_raw[len(Q_raw)//4] > 0 else "NEGATIVE"
axes[1].annotate(
    f"Rising-edge sign: {sign_rising}\n"
    f"(expected POSITIVE for η<0, α>0)",
    xy=(0.05, 0.05), xycoords="axes fraction", fontsize=8,
    color="darkgreen" if sign_rising == "POSITIVE" else "red",
)

plt.tight_layout()

# ── Figure 2: per-gate waveform (I & Q) with log-magnitude FFT below ──────────
# Each gate gets its own waveform panel (top row) and its own spectrum panel
# (bottom row). The spectrum is |FFT(I + iQ)| of the complex baseband envelope —
# the asymmetry DRAG creates to suppress spectral weight near the anharmonicity.
dt = 1e-9  # 1 GS/s sampling (1 ns per sample)
n_fft = 4096  # zero-pad for a smooth (sinc-interpolated) spectrum
freq_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt)) / 1e6

mags = {}
max_mag = 0.0
for name, wf in waveforms.items():
    z = wf["I"] + 1j * wf["Q"]
    m = np.abs(np.fft.fftshift(np.fft.fft(z, n_fft)))
    mags[name] = m
    max_mag = max(max_mag, m.max())

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7), sharex="row", sharey="row")
fig2.suptitle(
    "Gate waveforms (as built by config_helpers) and complex-envelope spectra\n"
    f"amp={AMP}, length={LENGTH_NS} ns, β={DRAG_BETA}, η={ANHARMONICITY/1e6:.0f} MHz"
    "  |  top: I = blue, Q = red (dashed)   bottom: |FFT(I+iQ)| log",
    fontsize=10,
)

for col, (name, wf) in enumerate(waveforms.items()):
    ax_t = axes2[0, col]
    ax_f = axes2[1, col]

    ax_t.plot(t, wf["I"], color="steelblue", label="I")
    ax_t.plot(t, wf["Q"], color="tomato", ls="--", label="Q")
    ax_t.axhline(0, color="k", lw=0.5)
    ax_t.set_title(name)
    ax_t.set_xlabel("sample (ns)")
    if col == 0:
        ax_t.set_ylabel("amplitude")
        ax_t.legend(fontsize=8)

    ax_f.plot(freq_MHz, mags[name], color="purple")
    ax_f.axvline(0, color="k", lw=0.5)
    ax_f.axvline(
        ANHARMONICITY / 1e6, color="gray", ls="--", lw=1,
        label=f"η = {ANHARMONICITY/1e6:.0f} MHz",
    )
    ax_f.set_yscale("log")
    ax_f.set_xlim(-400, 400)
    ax_f.set_xlabel("frequency (MHz)")
    if col == 0:
        ax_f.set_ylabel("|FFT(I+iQ)| (log)")
        ax_f.legend(fontsize=8)

axes2[1, 0].set_ylim(max_mag * 1e-4, max_mag * 1.5)  # shared across row via sharey

plt.tight_layout()

# ── Figure 3: DRAG Q signs summary ───────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
fig3.suptitle("DRAG derivative channel comparison across gates", fontsize=10)

colors = {"X90": "steelblue", "mX90": "tomato", "Y90": "seagreen", "mY90": "darkorange"}

# DRAG channel: Q for X gates, I for Y gates
drag_channels = {
    "X90":  waveforms["X90"]["Q"],
    "mX90": waveforms["mX90"]["Q"],
    "Y90":  waveforms["Y90"]["I"],
    "mY90": waveforms["mY90"]["I"],
}
for name, wf in drag_channels.items():
    ax3.plot(t, wf, label=name, color=colors[name])
ax3.axhline(0, color="k", lw=0.5)
ax3.set_xlabel("sample (ns)")
ax3.set_ylabel("DRAG derivative amplitude")
ax3.legend()

# ── Text summary ─────────────────────────────────────────────────────────────
print("\n── DRAG sign check ─────────────────────────────────────────────")
print(f"  raw[1] rising-edge sign : {sign_rising}")
print(f"  Expected (η<0, α>0)     : POSITIVE")
if sign_rising == "POSITIVE":
    print("  → qualang_tools sign is CORRECT.")
    print("    Config uses +raw[1] for X90_wf_der → DRAG applied in correct direction.")
    print("    Inverse gates use -raw[1] → consistent. Signs look good.")
else:
    print("  → qualang_tools sign is UNEXPECTED (negative when positive expected).")
    print("    Config uses +raw[1] for X90_wf_der → DRAG is BACKWARDS.")
    print("    Fix: negate all _wf_der_ waveforms in config_helpers.py")
    print("    (forward gates: -raw[1], inverse gates: +raw[1])")

print("\n  DRAG derivative channel rising-edge signs by gate:")
for name, wf in drag_channels.items():
    mid = len(wf) // 4
    sign = "+" if wf[mid] >= 0 else "-"
    print(f"    {name:6s}  DRAG ch rising-edge: {sign}  (val={wf[mid]:.4f})")
print("  X90 and mX90 should be opposite in sign.")
print("  Y90 and mY90 should be opposite in sign.")
print("────────────────────────────────────────────────────────────────\n")

plt.show()
