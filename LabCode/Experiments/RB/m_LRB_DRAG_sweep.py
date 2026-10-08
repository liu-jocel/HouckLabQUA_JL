from qm.qua import *
import numpy as np
import os
import time
import random as rd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import h5py

matplotlib.use("QtAgg")
plt.ion()

from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.ss_helpers import find_state_1q_nL
from LabCode.Helpers.qua_helpers import measure_1q
import lmfit
from LabCode.Helpers.RB_helpers import fit_leakage_RB, fit_L1_L2
from scipy.optimize import curve_fit
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def _fit_state2_lmfit(depths, p2):
    """Fit |2> population vs depths using lmfit. Returns (L1, L1_err, L2, L2_err, result)."""
    model = lmfit.Model(fit_L1_L2)
    params = model.make_params(L1=1e-5, L2=1e-3, e_spam=0.001)
    params['L1'].set(min=0, max=0.1)
    params['L2'].set(min=0, max=0.1)
    params['e_spam'].set(min=0, max=0.1)
    weights = 1 / np.sqrt(np.where(p2 > 0, p2, 1e-9))
    result = model.fit(p2, params, d=depths, weights=weights)
    L1 = result.params['L1'].value
    L2 = result.params['L2'].value
    L1_err = result.params['L1'].stderr or np.nan
    L2_err = result.params['L2'].stderr or np.nan
    return L1, L1_err, L2, L2_err, result


class RB1Q_Leakage_DRAGSweep(ExperimentClass):
    """
    Leakage RB swept over the DRAG alpha coefficient.

    For each alpha value, runs a short leakage RB (3-state classification,
    multiple circuit depths, several random circuits per depth). Fits the
    |2> population vs depth to extract the leakage rate L1, and plots
    L1 vs alpha live. The optimum DRAG alpha is the one that minimizes L1.

    Like CliffordDRAGCal, alpha is applied as a Q-channel scaling on
    every Clifford via the amp(...) matrix. Use this with a config
    built from a meta_dict where DRAG_beta = 1, so the Q waveform is
    actually non-zero before being scaled.

    Live plot: L1 vs alpha. Saved data: full populations(alpha, depth, 3)
    plus per-alpha fit results in HDF5.
    """

    def __init__(self, element, alpha_vec, circuit_depths, num_circuits, shots,
                 calib_params, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,
                         suffix=suffix, meta_dict=meta_dict, config=config)
        self.element = element
        self.config = config
        self.alpha_vec = np.asarray(alpha_vec, dtype=float)
        # Match RB1Q_Leakage convention: longest depth first
        self.circuit_depths = np.sort(np.asarray(circuit_depths, dtype=int))[::-1]
        self.max_depth = 2 * int(max(circuit_depths))
        self.num_circuits = int(num_circuits)
        self.shots = int(shots)
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1000000)
        self.resonator_dump_time = self.meta_dict.get('resonator_dump_time', 1000)

        self.data = {
            "qubit": self.element,
            "alpha_vec": self.alpha_vec,
            "circuit_depths": self.circuit_depths,
            "num_circuits": self.num_circuits,
            "shots": self.shots,
            "seed": self.seed,
            "meta_dict": self.meta_dict,
        }

        print(f"alpha sweep: {len(self.alpha_vec)} values")
        print(f"depths: {self.circuit_depths.tolist()}")
        print(f"circuits per alpha: {self.num_circuits}, shots per depth: {self.shots}")

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        try:
            self.analyze(self.job, self.qm)
        finally:
            self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            try:
                self._save_h5()
            except Exception as e:
                print(f"[RB1Q_Leakage_DRAGSweep] WARNING: h5 save failed: {e}")

    # ------------------------------------------------------------------
    # QUA program
    # ------------------------------------------------------------------
    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        therm = self.resonator_dump_time

        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            saved_gate = declare(int)
            a = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_ps_stream = declare_stream()
            Q_ps_stream = declare_stream()
            I_stream = declare_stream()
            Q_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                # Different sequences per k come from rand.rand_int(24) inside
                # generate_sequence; the seed argument is a Python int constant
                # (must fit in QUA's int representation, so don't add a QUA
                # variable to it).
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.max_depth)

                with for_each_(a, self.alpha_vec):
                    with for_each_(m, self.circuit_depths):
                        # Substitute the inverse so this depth is a closed sequence
                        assign(saved_gate, sequence_list[m])
                        assign(sequence_list[m], inv_gate_list[m - 1])

                        with for_(n, 0, n < self.shots, n + 1):
                            # ---- Pre-measurement for postselection ----
                            align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            save(I, I_ps_stream)
                            save(Q, Q_ps_stream)
                            wait(therm * u.ns)

                            # ---- RB sequence ----
                            align()
                            play_sequence_drag(self.element, sequence_list, m, a)

                            # ---- Result measurement ----
                            align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(therm * u.ns)

                        # Restore the original gate so the next depth substitution
                        # operates on the un-modified sequence
                        assign(sequence_list[m], saved_gate)

                save(k, k_stream)

            with stream_processing():
                k_stream.save("iteration")
                # Buffer order matches innermost-to-outermost of the QUA loops:
                # innermost shots, then depths, then alpha. save_all so we
                # can accumulate across circuit iterations on the client.
                I_ps_stream \
                    .buffer(self.shots) \
                    .buffer(len(self.circuit_depths)) \
                    .buffer(len(self.alpha_vec)) \
                    .save_all("I_ps")
                Q_ps_stream \
                    .buffer(self.shots) \
                    .buffer(len(self.circuit_depths)) \
                    .buffer(len(self.alpha_vec)) \
                    .save_all("Q_ps")
                I_stream \
                    .buffer(self.shots) \
                    .buffer(len(self.circuit_depths)) \
                    .buffer(len(self.alpha_vec)) \
                    .save_all("I")
                Q_stream \
                    .buffer(self.shots) \
                    .buffer(len(self.circuit_depths)) \
                    .buffer(len(self.alpha_vec)) \
                    .save_all("Q")

        return prog

    # ------------------------------------------------------------------
    # Analysis / live plotting
    # ------------------------------------------------------------------
    def analyze(self, job, qm):
        matplotlib.use("QtAgg")
        plt.ion()
        results = fetching_tool(
            job,
            data_list=["I_ps", "Q_ps", "I", "Q", "iteration"],
            mode="live",
        )

        n_alpha = len(self.alpha_vec)
        n_depths = len(self.circuit_depths)

        # Plot ascending depths for fit clarity
        depth_order = np.argsort(self.circuit_depths)
        depths_sorted = self.circuit_depths[depth_order]

        # ---------------- Figure ----------------
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Leakage RB DRAG sweep — {self.element}")
        interrupt_on_close(fig, job)

        L1_line, = ax.plot(self.alpha_vec, np.full(n_alpha, np.nan),
                           "o-", color="C3", lw=1, ms=5, label="L1 fit")
        min_marker, = ax.plot([], [], "o", color="black", ms=10,
                              zorder=5, label="min L1")
        ax.set_xlabel("DRAG alpha")
        ax.set_ylabel("L1 (leakage rate per Clifford)")
        ax.set_yscale("log")
        ax.set_xlim(self.alpha_vec.min(), self.alpha_vec.max())
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        # ---------- Second figure: RB curves at the best alpha ----------
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.suptitle(f"RB at best DRAG alpha — {self.element}")
        interrupt_on_close(fig2, job)

        state_colors = ["C0", "C1", "C2"]
        state_labels = ["|0⟩", "|1⟩", "|2⟩"]
        pop_lines = [
            ax2.plot(depths_sorted, np.zeros(n_depths), ".",
                     color=state_colors[k], label=state_labels[k])[0]
            for k in range(3)
        ]
        fit_lines = [
            ax2.plot([], [], "--", color=state_colors[k], alpha=0.7)[0]
            for k in range(3)
        ]
        ax2.set_xlabel("Circuit depth")
        ax2.set_ylabel("Population")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)
        ax2.legend(loc="center right", fontsize=9)
        best_alpha_text = ax2.text(
            0.02, 0.98, "",
            transform=ax2.transAxes, fontsize=9,
            ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7",
                      boxstyle="round,pad=0.3"),
        )
        # Dense x-grid for smooth fit curves
        depths_dense = np.linspace(
            max(depths_sorted.min(), 1), depths_sorted.max(), 200
        )

        # Per-alpha accumulators. Each entry is a list of (ss_chunk, valid_chunk)
        # tuples appended as new circuits arrive. Both chunks have shape
        # (samples, depths) where samples = new_circuits * shots.
        ss_buffers = [[] for _ in range(n_alpha)]
        populations = np.full((n_alpha, n_depths, 3), np.nan)
        kept_counts = np.zeros((n_alpha, n_depths), dtype=np.int64)
        L1_arr = np.full(n_alpha, np.nan)
        L1_err_arr = np.full(n_alpha, np.nan)
        L2_arr = np.full(n_alpha, np.nan)
        L2_err_arr = np.full(n_alpha, np.nan)
        fit_popts = [None] * n_alpha
        fit_pcovs = [None] * n_alpha
        e_spam_arr = np.full(n_alpha, np.nan)
        e_spam_err_arr = np.full(n_alpha, np.nan)
        fit_results = [None] * n_alpha
        ps_fraction_running = 0.0

        last_draw = time.time()
        draw_period = 0.3  # heavier per-tick work than DRAG cal (RB fits)
        last_processed = 0  # how many circuit-iterations we've already classified

        while results.is_processing():
            try:
                I_ps, Q_ps, I, Q, iteration = results.fetch_all()
            except Exception as e:
                print(f"\n[RB1Q_Leakage_DRAGSweep] fetch failed, exiting: {e}")
                break
            if any(x is None for x in (I_ps, Q_ps, I, Q)):
                continue

            I_ps = np.asarray(I_ps)
            Q_ps = np.asarray(Q_ps)
            I = np.asarray(I)
            Q = np.asarray(Q)
            if not (I_ps.shape == Q_ps.shape == I.shape == Q.shape):
                # Streams flush asynchronously — skip until they're aligned.
                # Truncate to the shortest first dim so we can use what we have.
                n_min = min(I_ps.shape[0], Q_ps.shape[0],
                            I.shape[0], Q.shape[0])
                if n_min == 0:
                    continue
                I_ps = I_ps[:n_min]
                Q_ps = Q_ps[:n_min]
                I = I[:n_min]
                Q = Q[:n_min]
            # Expected shape: (n_circuits_so_far, n_alpha, n_depths, n_shots)
            if I.ndim != 4 or I.shape[1:] != (n_alpha, n_depths, self.shots):
                continue

            n_circuits_so_far = I.shape[0]
            print(
                f"\rcircuit {n_circuits_so_far}/{self.num_circuits}"
                f" | kept: {ps_fraction_running:.3f}",
                end="", flush=True,
            )

            # Only classify the new circuits since the last refresh.
            if n_circuits_so_far > last_processed:
                I_ps_new = I_ps[last_processed:n_circuits_so_far]
                Q_ps_new = Q_ps[last_processed:n_circuits_so_far]
                I_new = I[last_processed:n_circuits_so_far]
                Q_new = Q[last_processed:n_circuits_so_far]
                # Shape (new_circuits, n_alpha, n_depths, n_shots)
                for ai in range(n_alpha):
                    I_ps_a = np.moveaxis(I_ps_new[:, ai], 0, 1).reshape(n_depths, -1)
                    Q_ps_a = np.moveaxis(Q_ps_new[:, ai], 0, 1).reshape(n_depths, -1)
                    I_a = np.moveaxis(I_new[:, ai], 0, 1).reshape(n_depths, -1)
                    Q_a = np.moveaxis(Q_new[:, ai], 0, 1).reshape(n_depths, -1)
                    ps_state = find_state_1q_nL(I_ps_a, Q_ps_a, self.calib_params)
                    res_state = find_state_1q_nL(I_a, Q_a, self.calib_params)
                    valid = ps_state == 0  # postselect on |0> pre-measurement
                    # Store transposed: (samples, depths)
                    ss_buffers[ai].append((res_state.T, valid.T))
                last_processed = n_circuits_so_far

            now = time.time()
            if now - last_draw > draw_period:
                # Compute postselected populations for each alpha, then fit
                total_kept = 0
                total_seen = 0
                for ai in range(n_alpha):
                    if not ss_buffers[ai]:
                        continue
                    if len(ss_buffers[ai]) == 1:
                        ss_arr, valid_arr = ss_buffers[ai][0]
                    else:
                        ss_arr = np.concatenate(
                            [c[0] for c in ss_buffers[ai]], axis=0
                        )
                        valid_arr = np.concatenate(
                            [c[1] for c in ss_buffers[ai]], axis=0
                        )

                    # counts[k, d] = # of kept shots that classified to state k
                    # at depth d. denom[d] = total kept shots at depth d.
                    counts = np.stack(
                        [((ss_arr == k) & valid_arr).sum(axis=0)
                         for k in (0, 1, 2)],
                        axis=0,
                    )
                    kept_at_depth = valid_arr.sum(axis=0).astype(np.int64)
                    denom = np.maximum(kept_at_depth, 1)
                    pop = counts / denom
                    populations[ai] = pop.T  # store as (depths, 3)
                    populations[ai, kept_at_depth == 0] = np.nan
                    kept_counts[ai] = kept_at_depth

                    total_kept += int(kept_at_depth.sum())
                    total_seen += int(valid_arr.size)

                    # Fit |2> trace if we have any signal
                    p2 = populations[ai, depth_order, 2]
                    if (np.any(p2 > 0) and np.all(np.isfinite(p2))
                            and np.all(kept_at_depth[depth_order] > 0)):
                        try:
                            L1, L1_err, L2, L2_err, res = _fit_state2_lmfit(depths_sorted, p2)
                            L1_arr[ai] = L1
                            L1_err_arr[ai] = L1_err
                            L2_arr[ai] = L2
                            L2_err_arr[ai] = L2_err
                            e_spam_arr[ai] = res.params['e_spam'].value
                            e_spam_err_arr[ai] = res.params['e_spam'].stderr or np.nan
                            fit_results[ai] = res
                        except Exception:
                            pass

                ps_fraction_running = (
                    total_kept / total_seen if total_seen else 0.0
                )

                # Update L1 line
                L1_line.set_ydata(L1_arr)
                if np.any(np.isfinite(L1_arr)):
                    finite = L1_arr[np.isfinite(L1_arr)]
                    positive = finite[finite > 0]
                    if positive.size:
                        y_lo = float(positive.min()) / 2.0
                        y_hi = float(positive.max()) * 2.0
                        ax.set_ylim(y_lo, y_hi)

                    best_idx = int(np.nanargmin(L1_arr))
                    min_marker.set_data(
                        [self.alpha_vec[best_idx]], [L1_arr[best_idx]]
                    )

                    # Update fig2: RB curves at the best alpha
                    best_pop = populations[best_idx]  # (depths, 3)
                    for k in range(3):
                        pop_lines[k].set_data(
                            depths_sorted,
                            best_pop[depth_order, k],
                        )
                        # Fit curve for this state
                        p_k = best_pop[depth_order, k]
                        if np.any(p_k > 0) and np.all(np.isfinite(p_k)):
                            try:
                                if k == 2:
                                    _, _, _, _, res_k = _fit_state2_lmfit(depths_sorted, p_k)
                                    fit_lines[k].set_data(depths_dense, res_k.eval(d=depths_dense))
                                else:
                                    popt_k, _ = curve_fit(
                                        fit_leakage_RB, depths_sorted, p_k,
                                        p0=[0.01, 0.005, 0.0], maxfev=2000,
                                    )
                                    fit_lines[k].set_data(
                                        depths_dense,
                                        fit_leakage_RB(depths_dense, *popt_k),
                                    )
                            except Exception:
                                fit_lines[k].set_data([], [])
                        else:
                            fit_lines[k].set_data([], [])

                    best_alpha_text.set_text(
                        f"α = {self.alpha_vec[best_idx]:.4f}\n"
                        f"L1 = {L1_arr[best_idx]:.4e}"
                    )
                    ax2.relim()
                    ax2.autoscale_view(scaley=False)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                fig2.canvas.draw_idle()
                fig2.canvas.flush_events()
                last_draw = now

        print("")
        try:
            qm.close()
        except Exception as e:
            print(f"[RB1Q_Leakage_DRAGSweep] WARNING: qm.close() failed: {e}")

        # Final pass: make sure everything is fitted on the latest data
        total_kept_final = 0
        total_seen_final = 0
        for ai in range(n_alpha):
            if not ss_buffers[ai]:
                continue
            if len(ss_buffers[ai]) == 1:
                ss_arr, valid_arr = ss_buffers[ai][0]
            else:
                ss_arr = np.concatenate(
                    [c[0] for c in ss_buffers[ai]], axis=0
                )
                valid_arr = np.concatenate(
                    [c[1] for c in ss_buffers[ai]], axis=0
                )
            counts = np.stack(
                [((ss_arr == k) & valid_arr).sum(axis=0) for k in (0, 1, 2)],
                axis=0,
            )
            kept_at_depth = valid_arr.sum(axis=0).astype(np.int64)
            denom = np.maximum(kept_at_depth, 1)
            pop = counts / denom
            populations[ai] = pop.T
            populations[ai, kept_at_depth == 0] = np.nan
            kept_counts[ai] = kept_at_depth

            total_kept_final += int(kept_at_depth.sum())
            total_seen_final += int(valid_arr.size)

            p2 = populations[ai, depth_order, 2]
            if (np.any(p2 > 0) and np.all(np.isfinite(p2))
                    and np.all(kept_at_depth[depth_order] > 0)):
                try:
                    L1, L1_err, L2, L2_err, res = _fit_state2_lmfit(depths_sorted, p2)
                    L1_arr[ai] = L1
                    L1_err_arr[ai] = L1_err
                    L2_arr[ai] = L2
                    L2_err_arr[ai] = L2_err
                    e_spam_arr[ai] = res.params['e_spam'].value
                    e_spam_err_arr[ai] = res.params['e_spam'].stderr or np.nan
                    fit_results[ai] = res
                except Exception:
                    pass

        ps_fraction_final = (
            total_kept_final / total_seen_final if total_seen_final else 0.0
        )

        # Best alpha = min finite L1
        if np.any(np.isfinite(L1_arr)):
            best_idx = int(np.nanargmin(L1_arr))
            best_alpha = float(self.alpha_vec[best_idx])
            best_L1 = float(L1_arr[best_idx])
        else:
            best_alpha = float("nan")
            best_L1 = float("nan")

        # Pack fit params as a uniform array (rows of NaN where fit failed): [L1, L2, e_spam]
        fit_params_arr = np.column_stack([L1_arr, L2_arr, e_spam_arr])  # (n_alpha, 3)

        self.data.update({
            "populations": populations,           # (alpha, depths, 3)
            "depth_order": depth_order,           # ascending order used in fits
            "L1": L1_arr,
            "L1_err": L1_err_arr,
            "L2": L2_arr,
            "L2_err": L2_err_arr,
            "e_spam": e_spam_arr,
            "e_spam_err": e_spam_err_arr,
            "fit_params": fit_params_arr,
            "kept_counts": kept_counts,
            "postselection_fraction": ps_fraction_final,
            "best_alpha": best_alpha,
            "best_L1": best_L1,
        })

        best_L2 = float(L2_arr[best_idx]) if np.isfinite(best_L1) else float("nan")
        best_lambda1 = 1 - best_L1 - best_L2 if np.isfinite(best_L1) else float("nan")
        print("=" * 60)
        print(f"Leakage RB DRAG sweep — {self.element}")
        print(f"  best alpha    : {best_alpha:.6f}")
        print(f"  best L1       : {best_L1:.6e}")
        print(f"  best L2       : {best_L2:.6e}")
        print(f"  best λ1       : {best_lambda1:.6f}")
        print(f"  fits done     : {int(np.sum(np.isfinite(L1_arr)))}/{n_alpha}")
        print(f"  postselection : {100 * ps_fraction_final:.2f}% kept")
        print("=" * 60)

        # Final render — defensive
        if plt.fignum_exists(fig.number):
            try:
                L1_line.set_ydata(L1_arr)
                if np.isfinite(best_L1):
                    min_marker.set_data([best_alpha], [best_L1])
                if np.any(np.isfinite(L1_arr)):
                    positive = L1_arr[(np.isfinite(L1_arr)) & (L1_arr > 0)]
                    if positive.size:
                        y_lo = float(positive.min()) / 2.0
                        y_hi = float(positive.max()) * 2.0
                        ax.set_ylim(y_lo, y_hi)
            except Exception as e:
                print(f"[RB1Q_Leakage_DRAGSweep] WARNING: final render fig1 failed: {e}")

        # Final render fig2 — best-alpha RB curves
        if np.isfinite(best_L1) and plt.fignum_exists(fig2.number):
            try:
                best_pop = populations[best_idx]
                for k in range(3):
                    pop_lines[k].set_data(
                        depths_sorted, best_pop[depth_order, k]
                    )
                    p_k = best_pop[depth_order, k]
                    if np.any(p_k > 0) and np.all(np.isfinite(p_k)):
                        try:
                            if k == 2:
                                _, _, _, _, res_k = _fit_state2_lmfit(depths_sorted, p_k)
                                fit_lines[k].set_data(depths_dense, res_k.eval(d=depths_dense))
                            else:
                                popt_k, _ = curve_fit(
                                    fit_leakage_RB, depths_sorted, p_k,
                                    p0=[0.01, 0.005, 0.0], maxfev=2000,
                                )
                                fit_lines[k].set_data(
                                    depths_dense,
                                    fit_leakage_RB(depths_dense, *popt_k),
                                )
                        except Exception:
                            fit_lines[k].set_data([], [])
                    else:
                        fit_lines[k].set_data([], [])
                best_alpha_text.set_text(
                    f"α = {best_alpha:.4f}\nL1 = {best_L1:.4e}"
                )
                ax2.relim()
                ax2.autoscale_view(scaley=False)
            except Exception as e:
                print(f"[RB1Q_Leakage_DRAGSweep] WARNING: final render fig2 failed: {e}")

        # Save both figures
        try:
            fig.savefig(self.iname, bbox_inches="tight")
        except Exception as e:
            print(f"[RB1Q_Leakage_DRAGSweep] WARNING: fig1 save failed: {e}")

        # fig2 saved next to fig with _RBcurves suffix
        try:
            base_img, ext_img = os.path.splitext(self.iname)
            fig2_path = base_img + "_RBcurves" + ext_img
            fig2.savefig(fig2_path, bbox_inches="tight")
        except Exception as e:
            print(f"[RB1Q_Leakage_DRAGSweep] WARNING: fig2 save failed: {e}")

        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            plt.close(fig2)
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
                RB1Q_Leakage_DRAGSweep._dict_to_h5(sub, val)
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


# ----------------------------------------------------------------------
# QUA helpers
# ----------------------------------------------------------------------
def generate_sequence(seed, longest_circuit):
    """
    Helper Function to create RB sequences (mirrors RB1Q_Leakage's helper).
    """
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=longest_circuit + 1)
    inv_gate = declare(int, size=longest_circuit + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < longest_circuit, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence_drag(element, sequence_list, depth, a):
    """
    Play an RB sequence with a DRAG-alpha scaling matrix on every Clifford.

    Mirrors CliffordDRAGCal's play_sequence: X-pulses scale Q by `a`,
    Y-pulses scale I by `a`. This requires a config built with
    DRAG_beta=1 so that the un-scaled Q waveforms are non-zero.
    """
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(4, element)
            with case_(1):
                play("X90" * amp(1, 0, 0, a), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(2):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("Y90" * amp(a, 0, 0, 1), element)
            with case_(3):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(4):
                play("X90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
            with case_(5):
                play("X90" * amp(1, 0, 0, a), element)
                play("mY90" * amp(a, 0, 0, 1), element)
            with case_(6):
                play("mX90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
            with case_(7):
                play("mX90" * amp(1, 0, 0, a), element)
                play("mY90" * amp(a, 0, 0, 1), element)
            with case_(8):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(9):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("mX90" * amp(1, 0, 0, a), element)
            with case_(10):
                play("mY90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(11):
                play("mY90" * amp(a, 0, 0, 1), element)
                play("mX90" * amp(1, 0, 0, a), element)
            with case_(12):
                play("X90" * amp(1, 0, 0, a), element)
            with case_(13):
                play("mX90" * amp(1, 0, 0, a), element)
            with case_(14):
                play("Y90" * amp(a, 0, 0, 1), element)
            with case_(15):
                play("mY90" * amp(a, 0, 0, 1), element)
            with case_(16):
                play("mX90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(17):
                play("mX90" * amp(1, 0, 0, a), element)
                play("mY90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(18):
                play("X90" * amp(1, 0, 0, a), element)
                play("X90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
            with case_(19):
                play("X90" * amp(1, 0, 0, a), element)
                play("X90" * amp(1, 0, 0, a), element)
                play("mY90" * amp(a, 0, 0, 1), element)
            with case_(20):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(21):
                play("Y90" * amp(a, 0, 0, 1), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("mX90" * amp(1, 0, 0, a), element)
            with case_(22):
                play("X90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("X90" * amp(1, 0, 0, a), element)
            with case_(23):
                play("mX90" * amp(1, 0, 0, a), element)
                play("Y90" * amp(a, 0, 0, 1), element)
                play("mX90" * amp(1, 0, 0, a), element)