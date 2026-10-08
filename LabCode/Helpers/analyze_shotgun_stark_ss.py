"""
Companion analysis script for ShotgunStarkSS HDF5 files.

Usage (CLI):
    python -m LabCode.Helpers.analyze_shotgun_stark_ss path/to/file.h5

Usage (import):
    from LabCode.Helpers.analyze_shotgun_stark_ss import analyze
    data, fig_grid, fig_sep = analyze("path/to/file.h5")
"""

from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────────

def load(fpath):
    """
    Load and parse a ShotgunStarkSS HDF5 file.

    Returns a dict with flat shot arrays and metadata.

    Keys
    ----
    I, Q       : (total_shots,)  demodulated quadratures
    amp        : (total_shots,)  Stark amplitude for each shot
    sign       : (total_shots,)  +1 / -1 detuning sign
    state      : (total_shots,)  0 = |g>, 1 = |e>
    unique_amps: sorted unique amplitude values
    unique_signs: [+1, -1] (positive first)
    cycles     : number of complete cycles collected
    N          : shots per cycle (number of (amp, sign, state) combos)
    attrs      : HDF5 root attributes dict
    """
    with h5py.File(fpath, "r") as f:
        I_raw     = f["I"][:]           # (cycles, N)
        Q_raw     = f["Q"][:]           # (cycles, N)
        ts_raw    = f["timestamps"][:]  # (cycles, N) OPX clock cycles (4 ns each)
        seq_amp   = f["seq_amp"][:]     # (N,)
        seq_sign  = f["seq_sign"][:].astype(int)  # (N,) cast from int8
        seq_state = f["seq_state"][:]   # (N,)
        attrs     = dict(f.attrs)

    cycles, N = I_raw.shape

    # Flatten: tile per-slot metadata across all cycles
    I_flat     = I_raw.ravel()
    Q_flat     = Q_raw.ravel()
    ts_flat    = ts_raw.ravel()
    amp_flat   = np.tile(seq_amp,   cycles)
    sign_flat  = np.tile(seq_sign,  cycles)
    state_flat = np.tile(seq_state, cycles)

    unique_amps  = np.sort(np.unique(seq_amp))
    unique_signs = np.sort(np.unique(seq_sign))[::-1]  # [+1, -1], derived from data

    return {
        "I":            I_flat,
        "Q":            Q_flat,
        "timestamps":   ts_flat,        # OPX clock cycles; multiply by 4e-9 for seconds
        "amp":          amp_flat,
        "sign":         sign_flat,
        "state":        state_flat,
        "unique_amps":  unique_amps,
        "unique_signs": unique_signs,
        "cycles":       cycles,
        "N":            N,
        "attrs":        attrs,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _sign_label(sgn, stark_det_hz):
    mhz = int(round(abs(stark_det_hz) / 1e6))
    return f"+{mhz} MHz" if sgn == 1 else f"−{mhz} MHz"


# ── Figure 1: IQ scatter grid ─────────────────────────────────────────────────

def plot_iq_grid(data, max_pts=2000, alpha=0.25, s=4):
    """
    One IQ scatter panel per (detuning sign, amplitude) pair.

    Rows = detuning sign, Cols = amplitude.
    Each panel is fully titled with both signed detuning and amplitude.
    """
    amps      = data["unique_amps"]
    signs     = data["unique_signs"]
    n_amp     = len(amps)
    n_sign    = len(signs)
    stark_det = float(data["attrs"].get("stark_detuning", 0))
    qubit     = data["attrs"].get("qubit", "")

    fig, axes = plt.subplots(
        n_sign, n_amp,
        figsize=(3.0 * n_amp, 3.0 * n_sign),
        squeeze=False,
    )
    fig.suptitle(
        f"{qubit}  Stark SS  ({data['cycles']} cycles)",
        fontsize=11,
    )

    rng = np.random.default_rng(0)

    for row, sgn in enumerate(signs):
        for col, amp in enumerate(amps):
            ax = axes[row, col]
            base_mask = (np.isclose(data["amp"], amp, atol=1e-9)) & (data["sign"] == sgn)

            for state, color, label in [(0, "steelblue", "|g⟩"), (1, "tomato", "|e⟩")]:
                m = base_mask & (data["state"] == state)
                Iv, Qv = data["I"][m], data["Q"][m]
                if len(Iv) > max_pts:
                    idx = rng.choice(len(Iv), max_pts, replace=False)
                    Iv, Qv = Iv[idx], Qv[idx]
                ax.scatter(Iv, Qv, s=s, alpha=alpha, color=color,
                           label=label, rasterized=True)

            ax.set_aspect("equal")
            ax.set_title(f"Δ = {_sign_label(sgn, stark_det)},  a = {amp:.3f}", fontsize=9)
            ax.set_xlabel("I  (a.u.)", fontsize=8)
            ax.set_ylabel("Q  (a.u.)", fontsize=8)
            if row == 0 and col == n_amp - 1:
                ax.legend(fontsize=7, markerscale=3, framealpha=0.7)

    plt.tight_layout()
    return fig


# ── Figure 2: centroid separation vs amplitude ────────────────────────────────

def plot_separation(data):
    """
    |g>–|e> IQ centroid separation vs Stark amplitude,
    one curve per detuning sign.
    """
    amps      = data["unique_amps"]
    signs     = data["unique_signs"]
    stark_det = float(data["attrs"].get("stark_detuning", 0))
    qubit     = data["attrs"].get("qubit", "")

    colors = {1: "steelblue", -1: "tomato"}

    fig, ax = plt.subplots(figsize=(6, 4))

    for sgn in signs:
        seps = []
        for amp in amps:
            m  = (np.isclose(data["amp"], amp, atol=1e-9)) & (data["sign"] == sgn)
            mg = m & (data["state"] == 0)
            me = m & (data["state"] == 1)
            if mg.sum() < 2 or me.sum() < 2:
                seps.append(np.nan)
                continue
            cg = np.array([data["I"][mg].mean(), data["Q"][mg].mean()])
            ce = np.array([data["I"][me].mean(), data["Q"][me].mean()])
            seps.append(float(np.linalg.norm(ce - cg)))
        ax.plot(amps, seps, marker="o", color=colors[sgn],
                label=_sign_label(sgn, stark_det))

    ax.set_xlabel("Stark amplitude (a.u.)")
    ax.set_ylabel("|g⟩–|e⟩ centroid separation (a.u.)")
    ax.set_title(f"{qubit}  IQ separation vs Stark amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Figure 3: 1-D I histograms (projection along g–e axis) ───────────────────

def plot_histograms(data, bins=60):
    """
    One projected histogram panel per (detuning sign, amplitude) pair.

    Projects each shot onto the g→e centroid axis.
    Each panel is fully titled with both signed detuning and amplitude.
    """
    amps      = data["unique_amps"]
    signs     = data["unique_signs"]
    stark_det = float(data["attrs"].get("stark_detuning", 0))
    qubit     = data["attrs"].get("qubit", "")

    n_amp  = len(amps)
    n_sign = len(signs)

    fig, axes = plt.subplots(
        n_sign, n_amp,
        figsize=(3.0 * n_amp, 2.8 * n_sign),
        squeeze=False,
    )
    fig.suptitle(f"{qubit}  Projected histograms", fontsize=11)

    for row, sgn in enumerate(signs):
        for col, amp in enumerate(amps):
            ax = axes[row, col]
            base_mask = (np.isclose(data["amp"], amp, atol=1e-9)) & (data["sign"] == sgn)

            Ig = data["I"][base_mask & (data["state"] == 0)]
            Ie = data["I"][base_mask & (data["state"] == 1)]
            Qg = data["Q"][base_mask & (data["state"] == 0)]
            Qe = data["Q"][base_mask & (data["state"] == 1)]

            cg = np.array([Ig.mean(), Qg.mean()]) if len(Ig) > 1 else np.zeros(2)
            ce = np.array([Ie.mean(), Qe.mean()]) if len(Ie) > 1 else np.ones(2)
            axis = ce - cg
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 1e-30 else np.array([1.0, 0.0])

            proj_g = Ig * axis[0] + Qg * axis[1]
            proj_e = Ie * axis[0] + Qe * axis[1]

            all_vals = np.concatenate([proj_g, proj_e])
            edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

            ax.hist(proj_g, bins=edges, color="steelblue", alpha=0.6,
                    density=True, label="|g⟩")
            ax.hist(proj_e, bins=edges, color="tomato",    alpha=0.6,
                    density=True, label="|e⟩")

            ax.set_title(f"Δ = {_sign_label(sgn, stark_det)},  a = {amp:.3f}", fontsize=9)
            ax.set_xlabel("projected quadrature  (a.u.)", fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            if row == 0 and col == n_amp - 1:
                ax.legend(fontsize=7, framealpha=0.7)

    plt.tight_layout()
    return fig


# ── Figure 4: sliding-window IQ movie ────────────────────────────────────────

def animate_sliding_window(data, amp, sgn, window=100, step=1, interval=50):
    """
    Animate the |g⟩ IQ cloud for one (amp, sign) condition as a sliding
    window of `window` consecutive shots, ordered by hardware timestamp.

    The title updates each frame with the median elapsed time of the window.

    Parameters
    ----------
    data     : dict returned by load()
    amp      : Stark amplitude to plot (must match a value in data["unique_amps"])
    sgn      : detuning sign (+1 or -1)
    window   : number of shots per frame
    interval : delay between frames in milliseconds

    Returns
    -------
    fig, anim   (keep a reference to anim or it will be garbage-collected)
    """
    from matplotlib.animation import FuncAnimation

    stark_det = float(data["attrs"].get("stark_detuning", 0))
    qubit     = data["attrs"].get("qubit", "")
    slabel    = _sign_label(sgn, stark_det)

    def _sorted_state(state_val):
        mask = (
            np.isclose(data["amp"], amp, atol=1e-9)
            & (data["sign"] == sgn)
            & (data["state"] == state_val)
        )
        Iv = data["I"][mask]
        Qv = data["Q"][mask]
        ts = data["timestamps"][mask]
        order = np.argsort(ts)
        return Iv[order], Qv[order], ts[order]

    Ig, Qg, tsg = _sorted_state(0)
    Ie, Qe, tse = _sorted_state(1)

    # Reference time: earliest timestamp across both states
    t0 = min(tsg[0], tse[0])
    tg_elapsed = (tsg - t0) * 4e-9   # seconds
    te_elapsed = (tse - t0) * 4e-9

    n_shots  = min(len(Ig), len(Ie))
    n_frames = max(1, (n_shots - window) // step + 1)

    # Shared axis limits covering both states
    I_all = np.concatenate([Ig, Ie])
    Q_all = np.concatenate([Qg, Qe])
    pad   = 0.05 * max(I_all.ptp(), Q_all.ptp())
    xlim  = (I_all.min() - pad, I_all.max() + pad)
    ylim  = (Q_all.min() - pad, Q_all.max() + pad)

    fig, (ax_g, ax_e) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{qubit}   Δ = {slabel},  a = {amp:.3f}", fontsize=11)

    sc_g = ax_g.scatter([], [], s=6, alpha=0.4, color="steelblue", rasterized=True)
    sc_e = ax_e.scatter([], [], s=6, alpha=0.4, color="tomato",    rasterized=True)

    for ax, label in [(ax_g, "|g⟩"), (ax_e, "|e⟩")]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("I  (a.u.)")
        ax.set_ylabel("Q  (a.u.)")
        ax.set_title(label)

    def _update(frame):
        start = frame * step
        sc_g.set_offsets(np.column_stack([Ig[start: start + window],
                                          Qg[start: start + window]]))
        sc_e.set_offsets(np.column_stack([Ie[start: start + window],
                                          Qe[start: start + window]]))
        t_med = 0.5 * (tg_elapsed[start + window // 2]
                       + te_elapsed[start + window // 2])
        fig.suptitle(
            f"{qubit}   Δ = {slabel},  a = {amp:.3f}"
            f"     shots {start}–{start + window - 1}"
            f"     t_median = {t_med:.2f} s",
            fontsize=10,
        )

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=interval, blit=False)
    return fig, anim


# ── Top-level entry point ─────────────────────────────────────────────────────

def analyze(fpath, show=True, save=True):
    """
    Load a ShotgunStarkSS HDF5 file and produce three figures:

    1. IQ scatter grid  (per amp × detuning sign)
    2. Centroid separation vs amplitude
    3. Projected histogram grid

    Parameters
    ----------
    fpath : str or Path
    show  : bool  — call plt.show() after generating
    save  : bool  — save PNGs alongside the source file

    Returns
    -------
    data, fig_grid, fig_sep, fig_hist
    """
    fpath = Path(fpath)
    data  = load(fpath)

    print(f"File   : {fpath.name}")
    print(f"Qubit  : {data['attrs'].get('qubit', '?')}")
    print(f"Cycles : {data['cycles']}")
    print(f"N/cycle: {data['N']}")
    print(f"Amps   : {data['unique_amps']}")
    print(f"Detuning: ±{int(round(abs(float(data['attrs'].get('stark_detuning',0)))/1e6))} MHz")

    fig_grid = plot_iq_grid(data)
    fig_sep  = plot_separation(data)
    fig_hist = plot_histograms(data)

    if save:
        stem = fpath.with_suffix("")
        for fig, tag in [(fig_grid, "_iq_grid"),
                         (fig_sep,  "_separation"),
                         (fig_hist, "_histograms")]:
            out = str(stem) + tag + ".png"
            fig.savefig(out, dpi=150)
            print(f"  saved → {Path(out).name}")

    if show:
        plt.show()

    return data, fig_grid, fig_sep, fig_hist


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("QtAgg")

    # ── Set this to your .h5 file path ───────────────────────────────────────
    FILE = r"z:\t1Team\Data\2026-04-27_BFG_cooldown\TATCL02\OPX\q1\q1_2026_05_14\q1_18_25_52_stark_ss.h5"
    # ─────────────────────────────────────────────────────────────────────────

    ANIMATE        = True    # set False to skip the movie
    ANIMATE_AMP    = 0.1     # Stark amplitude to animate
    ANIMATE_SIGN   = 1       # +1 or -1
    ANIMATE_WINDOW = 500     # sliding window width in shots
    ANIMATE_STEP   = 3       # shots to advance per frame

    data, fig_grid, fig_sep, fig_hist = analyze(FILE, show=False)

    if ANIMATE:
        fig_anim, anim = animate_sliding_window(
            data,
            amp=ANIMATE_AMP,
            sgn=ANIMATE_SIGN,
            window=ANIMATE_WINDOW,
            step=ANIMATE_STEP,
        )

    plt.show()
