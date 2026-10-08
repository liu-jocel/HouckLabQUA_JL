from qm.qua import *
import numpy as np
import os
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import h5py

matplotlib.use("QtAgg")
plt.ion()

from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import find_state_1q_nL
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


class DragCal1Q(ExperimentClass):
    """
    DRAG Calibration with 3-state readout, postselection, and live
    plotting (ALE-style incremental accumulation).

    Each iteration: pre-measurement -> wait -> error-amplifying pulse
    train -> result measurement -> wait. Only shots whose pre-measurement
    classifies as |0> are kept.

    Streaming: I_ps, Q_ps, I, Q are streamed as one (num_pi, alpha)
    buffer per shot via `.save("...")` (not save_all). Each fetch
    returns the latest single buffer; we classify that batch and fold
    its counts into running accumulators. This keeps per-tick work
    bounded regardless of how long the run has been going, which is
    what stops the live plot from getting progressively slower and
    eventually freezing.
    """

    def __init__(self, element, alpha_vec, pi_list, shots, calib_params,
                 pulse_type="X180", outer_folder='', path='Rabi',
                 suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,
                         suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.pulse_type = pulse_type
        self.X180_amp_norm = self.meta_dict['X180']['amp']
        self.X90_amp_norm = self.meta_dict['X90']['amp']
        self.alpha_vec = alpha_vec

        self.shots = shots
        self.num_pi = np.array(pi_list, dtype=int)
        self.calib_params = calib_params

        self.read_len = self.meta_dict['global_params']['read_len']
        self.resonator_dump_time = self.meta_dict.get('resonator_dump_time', 1000)

        self.data = {
            'qubit': self.element,
            "pulse_type": self.pulse_type,
            "number_pulses": self.num_pi,
            "shots": self.shots,
            "alpha_vec": self.alpha_vec,
            "meta_dict": self.meta_dict,
        }

        print(len(self.alpha_vec))
        print(len(self.num_pi))

        self.config = config
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        try:
            self.analyze(self.job, self.qm)
        finally:
            # Always save, even if the user closed the figure mid-run
            # or analyze() raised partway through.
            self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            try:
                self._save_h5()
            except Exception as e:
                print(f"[DragCal1Q] WARNING: h5 save failed: {e}")

    # ------------------------------------------------------------------
    # QUA program
    # ------------------------------------------------------------------
    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        shots = self.shots
        therm = self.resonator_dump_time

        pulse_name = self.pulse_type  # "X180" or "X90"

        with program() as prog:
            n = declare(int)
            a = declare(fixed)
            m = declare(int)
            pi_pulses = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_ps_stream = declare_stream()
            Q_ps_stream = declare_stream()
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(pi_pulses, self.num_pi):
                    with for_(*from_array(a, self.alpha_vec)):
                        # Pre-measurement
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_ps_stream)
                        save(Q, Q_ps_stream)
                        wait(therm * u.ns)

                        # Pulse train
                        align(qubit, resonator)
                        with for_(m, 0, m < pi_pulses, m + 1):
                            play(pulse_name * amp(1, 0, 0, a), qubit)
                            play(pulse_name * amp(-1, 0, 0, -a), qubit)

                        # Result measurement
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(therm * u.ns)

            with stream_processing():
                # ALE-style: one (N, alpha) buffer per shot, NOT save_all.
                # Each fetch returns just the latest single shot's data.
                I_ps_stream.buffer(len(self.alpha_vec)).buffer(len(self.num_pi)).save("I_ps")
                Q_ps_stream.buffer(len(self.alpha_vec)).buffer(len(self.num_pi)).save("Q_ps")
                I_stream.buffer(len(self.alpha_vec)).buffer(len(self.num_pi)).save("I")
                Q_stream.buffer(len(self.alpha_vec)).buffer(len(self.num_pi)).save("Q")

        return prog

    # ------------------------------------------------------------------
    # Analysis dispatch
    # ------------------------------------------------------------------
    def analyze(self, job, qm):
        if len(self.num_pi) == 1:
            self._analyze_scatter(job, qm)
        else:
            self._analyze_heatmap(job, qm)

    # ------------------------------------------------------------------
    # Live loops (ALE pattern)
    # ------------------------------------------------------------------
    def _analyze_heatmap(self, job, qm):
        result = fetching_tool(
            job,
            data_list=["I_ps", "Q_ps", "I", "Q"],
            mode="live",
        )

        length_alpha = len(self.alpha_vec)
        length_N = len(self.num_pi)
        alpha_vals = np.asarray(self.alpha_vec, dtype=float)

        # ---------------- Figure ----------------
        fig, ax_img = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"DRAG Calibration — {self.element} ({self.pulse_type})")
        interrupt_on_close(fig, job)

        data = np.zeros((length_N, length_alpha))
        im = ax_img.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        cbar = plt.colorbar(im, ax=ax_img)
        cbar.set_label("|2⟩ population (postselected)")
        ax_img.set_ylabel("Number of pulse pairs N")
        ax_img.set_xlabel("DRAG alpha")

        num_ticks = min(5, length_alpha)
        alpha_tick_idx = np.linspace(0, length_alpha - 1, num_ticks, dtype=int)
        ax_img.set_xticks(alpha_tick_idx)
        ax_img.set_xticklabels([f"{alpha_vals[i]:.3f}" for i in alpha_tick_idx])

        num_ticks_N = min(5, length_N)
        N_tick_idx = np.linspace(0, length_N - 1, num_ticks_N, dtype=int)
        ax_img.set_yticks(N_tick_idx)
        ax_img.set_yticklabels([str(self.num_pi[i]) for i in N_tick_idx])

        # ---------------- Incremental accumulators ----------------
        # counts[k, N, alpha] = number of kept shots that classified to state k
        # at this (N, alpha). kept_total[N, alpha] = number of valid (postselected)
        # shots at this (N, alpha). Both grow shot-by-shot.
        counts = np.zeros((3, length_N, length_alpha), dtype=np.int64)
        kept_total = np.zeros((length_N, length_alpha), dtype=np.int64)
        seen_total = 0
        kept_total_scalar = 0

        last_draw = time.time()
        draw_period = 0.2

        while result.is_processing():
            fetched = result.fetch_all()
            if fetched is None:
                continue
            I_ps, Q_ps, I, Q = fetched
            if any(x is None for x in (I_ps, Q_ps, I, Q)):
                continue

            I_ps = np.asarray(I_ps)
            Q_ps = np.asarray(Q_ps)
            I = np.asarray(I)
            Q = np.asarray(Q)

            # In mode="live" with .save(...) buffered as (N, alpha),
            # each fetch returns one shot's worth: shape (N, alpha).
            if I.shape != (length_N, length_alpha):
                continue
            if I_ps.shape != I.shape or Q_ps.shape != I.shape or Q.shape != I.shape:
                continue

            # Classify just this single shot's batch
            ps_state = find_state_1q_nL(
                I_ps.reshape(-1)[None, :],
                Q_ps.reshape(-1)[None, :],
                self.calib_params,
            )[0].reshape(length_N, length_alpha)
            res_state = find_state_1q_nL(
                I.reshape(-1)[None, :],
                Q.reshape(-1)[None, :],
                self.calib_params,
            )[0].reshape(length_N, length_alpha)

            valid = ps_state == 0
            kept_total += valid
            for k in (0, 1, 2):
                counts[k] += (res_state == k) & valid
            seen_total += 1
            kept_total_scalar += int(valid.sum())

            ps_fraction = kept_total_scalar / max(seen_total * length_N * length_alpha, 1)

            print(
                f"\r{seen_total}/{self.shots} | kept: {ps_fraction:.3f}",
                end="",
                flush=True,
            )

            now = time.time()
            if now - last_draw > draw_period:
                denom = np.maximum(kept_total, 1)
                p2 = counts[2] / denom
                # Cells with no kept shots → NaN, displayed as 0
                p2 = np.where(kept_total == 0, np.nan, p2)

                data[:] = np.where(np.isnan(p2), 0.0, p2)
                im.set_data(data)
                finite = p2[np.isfinite(p2)]
                if finite.size:
                    im.set_clim(vmin=float(np.nanmin(finite)),
                                vmax=float(np.nanmax(finite)))

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_draw = now

        print("")
        qm.close()

        # Final summary
        denom = np.maximum(kept_total, 1)
        populations = np.zeros((length_N, length_alpha, 3), dtype=float)
        for k in (0, 1, 2):
            populations[..., k] = counts[k] / denom
        populations[kept_total == 0] = np.nan

        p2 = populations[..., 2]
        deepest_slice = p2[-1, :]
        if np.any(np.isfinite(deepest_slice)):
            best_idx = int(np.nanargmin(deepest_slice))
            best_alpha = float(alpha_vals[best_idx])
            best_p2 = float(deepest_slice[best_idx])
        else:
            best_alpha = float("nan")
            best_p2 = float("nan")

        ps_fraction_final = (
            kept_total_scalar / max(seen_total * length_N * length_alpha, 1)
        )

        self.data.update({
            "counts": counts,
            "kept_counts": kept_total,
            "populations": populations,
            "p2": p2,
            "shots_processed": seen_total,
            "postselection_fraction": ps_fraction_final,
            "best_alpha": best_alpha,
            "best_alpha_p2": best_p2,
        })

        N_at_best = int(self.num_pi[-1])
        print("=" * 60)
        print(f"DRAG calibration result — {self.element} ({self.pulse_type})")
        print(f"  best alpha       : {best_alpha:.6f}")
        print(f"  P(|2>) at best   : {best_p2:.4f}    (at N = {N_at_best})")
        print(f"  postselection    : {100 * ps_fraction_final:.2f}% kept")
        print(f"  shots processed  : {seen_total}/{self.shots}")
        print("=" * 60)

        # Final static render + save. If the user closed the figure
        # mid-run, plt.fignum_exists() returns False and we skip the
        # canvas-touching ops to avoid backend exceptions; we still
        # try to write the figure to disk because savefig works on
        # the Figure object itself even after the GUI window is gone.
        if plt.fignum_exists(fig.number):
            try:
                data[:] = np.where(np.isnan(p2), 0.0, p2)
                im.set_data(data)
                finite = p2[np.isfinite(p2)]
                if finite.size:
                    im.set_clim(vmin=float(np.nanmin(finite)),
                                vmax=float(np.nanmax(finite)))
            except Exception as e:
                print(f"[DragCal1Q] WARNING: final render failed: {e}")
        try:
            fig.savefig(self.iname, bbox_inches="tight")
        except Exception as e:
            print(f"[DragCal1Q] WARNING: figure save failed: {e}")
        try:
            plt.close(fig)
        except Exception:
            pass

    def _analyze_scatter(self, job, qm):
        result = fetching_tool(
            job,
            data_list=["I_ps", "Q_ps", "I", "Q"],
            mode="live",
        )

        length_alpha = len(self.alpha_vec)
        length_N = 1
        alpha_vals = np.asarray(self.alpha_vec, dtype=float)
        N_val = int(self.num_pi[0])

        # ---------------- Figure ----------------
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            f"DRAG Calibration — {self.element} ({self.pulse_type})  N = {N_val}"
        )
        interrupt_on_close(fig, job)

        scatter = ax.scatter(
            alpha_vals,
            np.zeros_like(alpha_vals),
            c="C0",
            s=25,
            zorder=3,
            label="|2⟩",
        )
        line, = ax.plot(alpha_vals, np.zeros_like(alpha_vals),
                        color="C0", lw=1, alpha=0.5, zorder=2)
        min_marker, = ax.plot([], [], "o", color="red", ms=10,
                              zorder=4, label="min")
        ax.set_xlabel("DRAG alpha")
        ax.set_ylabel("|2⟩ population (postselected)")
        ax.set_xlim(alpha_vals.min(), alpha_vals.max())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        # ---------------- Incremental accumulators ----------------
        counts = np.zeros((3, length_N, length_alpha), dtype=np.int64)
        kept_total = np.zeros((length_N, length_alpha), dtype=np.int64)
        seen_total = 0
        kept_total_scalar = 0

        last_draw = time.time()
        draw_period = 0.2

        while result.is_processing():
            fetched = result.fetch_all()
            if fetched is None:
                continue
            I_ps, Q_ps, I, Q = fetched
            if any(x is None for x in (I_ps, Q_ps, I, Q)):
                continue

            I_ps = np.asarray(I_ps)
            Q_ps = np.asarray(Q_ps)
            I = np.asarray(I)
            Q = np.asarray(Q)

            if I.shape != (length_N, length_alpha):
                continue
            if I_ps.shape != I.shape or Q_ps.shape != I.shape or Q.shape != I.shape:
                continue

            ps_state = find_state_1q_nL(
                I_ps.reshape(-1)[None, :],
                Q_ps.reshape(-1)[None, :],
                self.calib_params,
            )[0].reshape(length_N, length_alpha)
            res_state = find_state_1q_nL(
                I.reshape(-1)[None, :],
                Q.reshape(-1)[None, :],
                self.calib_params,
            )[0].reshape(length_N, length_alpha)

            valid = ps_state == 0
            kept_total += valid
            for k in (0, 1, 2):
                counts[k] += (res_state == k) & valid
            seen_total += 1
            kept_total_scalar += int(valid.sum())

            ps_fraction = kept_total_scalar / max(seen_total * length_alpha, 1)

            print(
                f"\r{seen_total}/{self.shots} | kept: {ps_fraction:.3f}",
                end="",
                flush=True,
            )

            now = time.time()
            if now - last_draw > draw_period:
                denom = np.maximum(kept_total, 1)
                p2 = counts[2] / denom
                p2 = np.where(kept_total == 0, np.nan, p2)
                trace = p2[0]
                trace_safe = np.where(np.isnan(trace), 0.0, trace)

                scatter.set_offsets(np.column_stack([alpha_vals, trace_safe]))
                line.set_ydata(trace_safe)

                if np.any(np.isfinite(trace)):
                    best_idx = int(np.nanargmin(trace))
                    min_marker.set_data([alpha_vals[best_idx]],
                                        [trace[best_idx]])
                    finite = trace[np.isfinite(trace)]
                    y_lo = max(0.0, float(finite.min()) - 0.05)
                    y_hi = min(1.0, float(finite.max()) + 0.05)
                    if y_hi > y_lo:
                        ax.set_ylim(y_lo, y_hi)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_draw = now

        print("")
        qm.close()

        denom = np.maximum(kept_total, 1)
        populations = np.zeros((length_N, length_alpha, 3), dtype=float)
        for k in (0, 1, 2):
            populations[..., k] = counts[k] / denom
        populations[kept_total == 0] = np.nan

        trace = populations[0, :, 2]
        if np.any(np.isfinite(trace)):
            best_idx = int(np.nanargmin(trace))
            best_alpha = float(alpha_vals[best_idx])
            best_p2 = float(trace[best_idx])
        else:
            best_alpha = float("nan")
            best_p2 = float("nan")

        ps_fraction_final = (
            kept_total_scalar / max(seen_total * length_alpha, 1)
        )

        self.data.update({
            "counts": counts,
            "kept_counts": kept_total,
            "populations": populations,
            "p2": populations[..., 2],
            "shots_processed": seen_total,
            "postselection_fraction": ps_fraction_final,
            "best_alpha": best_alpha,
            "best_alpha_p2": best_p2,
        })

        print("=" * 60)
        print(f"DRAG calibration result — {self.element} ({self.pulse_type})")
        print(f"  best alpha       : {best_alpha:.6f}")
        print(f"  P(|2>) at best   : {best_p2:.4f}    (at N = {N_val})")
        print(f"  postselection    : {100 * ps_fraction_final:.2f}% kept")
        print(f"  shots processed  : {seen_total}/{self.shots}")
        print("=" * 60)

        # Final static render + save (defensive — see heatmap version)
        if plt.fignum_exists(fig.number):
            try:
                trace_safe = np.where(np.isnan(trace), 0.0, trace)
                scatter.set_offsets(np.column_stack([alpha_vals, trace_safe]))
                line.set_ydata(trace_safe)
                if np.isfinite(best_p2):
                    min_marker.set_data([best_alpha], [best_p2])
            except Exception as e:
                print(f"[DragCal1Q] WARNING: final render failed: {e}")
        try:
            fig.savefig(self.iname, bbox_inches="tight")
        except Exception as e:
            print(f"[DragCal1Q] WARNING: figure save failed: {e}")
        try:
            plt.close(fig)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HDF5 saving
    # ------------------------------------------------------------------
    def _save_h5(self):
        base, _ = os.path.splitext(self.fname)
        h5_path = base + ".h5"

        print(f"Saving {h5_path}")
        with h5py.File(h5_path, "w") as f:
            self._dict_to_h5(f, self.data)

        self.h5_path = h5_path

    @staticmethod
    def _dict_to_h5(group, d):
        for key, val in d.items():
            key = str(key)
            if isinstance(val, dict):
                sub = group.create_group(key)
                DragCal1Q._dict_to_h5(sub, val)
            elif val is None:
                group.attrs[key] = "None"
            elif isinstance(val, (str, bytes)):
                group.attrs[key] = val
            elif isinstance(val, (int, float, np.integer, np.floating, bool, np.bool_)):
                group.attrs[key] = val
            else:
                try:
                    arr = np.asarray(val)
                    if arr.dtype.kind in ("U", "O"):
                        arr = np.asarray(
                            [str(x) for x in arr.ravel()]
                        ).reshape(arr.shape)
                        group.create_dataset(
                            key, data=arr.astype(h5py.string_dtype())
                        )
                    else:
                        group.create_dataset(key, data=arr)
                except Exception as e:
                    group.attrs[key] = (
                        f"<unserializable: {type(val).__name__}: {e}>"
                    )

    def save_data(self, data=None):
        print(f'Saving {self.fname} (as .h5)')
        if data is None:
            payload = self.data
        else:
            payload = data.get('data', data)
        base, _ = os.path.splitext(self.fname)
        h5_path = base + ".h5"
        with h5py.File(h5_path, "w") as f:
            self._dict_to_h5(f, payload)