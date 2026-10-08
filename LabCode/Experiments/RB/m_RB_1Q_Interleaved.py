from qm.qua import *
import numpy as np
import h5py
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Experiments.RB.m_RB_1Q import RB1Q, play_sequence
from LabCode.Experiments.Gates.m_ramsey_frequency import RamseyFrequency
from scipy.optimize import curve_fit
import random as rd
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

_D = 2  # qubit Hilbert-space dimension


class RB1Q_Interleaved(ExperimentClass):
    """
    Full IRB suite.

    On instantiation:
      1. Runs reference RB (standard RB1Q).
      2. Runs IRB for each gate in `gates`.
      3. Computes gate fidelities via the Magesan formula.
      4. Saves per-gate figures, a summary figure, and HDF5.

    Default gates: X90, mX90, Y90, mY90, I (16 ns identity wait).

    Gate fidelity formula (Magesan et al.):
        r_gate = (d-1)/d * (1 - p_irb / p_ref),  F_gate = 1 - r_gate
    where d=2, p_ref and p_irb are the exponential decay parameters.
    """

    DEFAULT_GATES = ["X90", "mX90", "Y90", "mY90", "I"]

    def __init__(
        self,
        element,
        circuit_depths,
        num_circuits,
        shots,
        calib_params,
        outer_folder="",
        path="",
        suffix="data",
        meta_dict=None,
        config=None,
        gates=None,
        bare_frequency=None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            outer_folder=outer_folder,
            suffix=suffix,
            meta_dict=meta_dict,
            config=config,
            **kwargs,
        )

        self.element = element
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.gates = gates if gates is not None else self.DEFAULT_GATES
        self.max_depth = max(circuit_depths)

        # IRB depths are half the reference depths so total gate count matches:
        # m random Cliffords interleaved with m gates = 2m operations ~ 1 ref step.
        self.irb_circuit_depths = list(
            dict.fromkeys(max(1, d // 2) for d in circuit_depths)
        )
        self.irb_max_depth = max(self.irb_circuit_depths)

        self.data = {
            "meta_dict": meta_dict,
            "config": config,
            "circuit_depths": circuit_depths,
            "irb_circuit_depths": self.irb_circuit_depths,
            "num_circuits": num_circuits,
            "shots": shots,
            "gates": self.gates,
        }

        # ── 0. Ramsey: measure bare qubit frequency ──────────────────────────
        if bare_frequency is None:
            print("=" * 60)
            print(f"Running Ramsey frequency measurement on {element}")
            print("=" * 60)
            t_vec = np.arange(16, 2000, 32)  # 16 ns to 2 us in 32 ns steps
            ramsey = RamseyFrequency(
                element,
                t_vec,
                shots=400,
                detuning=1 * u.MHz,
                calib_params=calib_params,
                outer_folder=outer_folder,
                path=path,
                suffix=suffix + "_ramsey",
                meta_dict=meta_dict,
                config=config,
            )
            if ramsey.data.get("delta") is not None:
                bare_frequency = int(meta_dict["IF_freq"] + ramsey.data["delta"][0] * u.MHz)
                print(f"Bare frequency: {bare_frequency / u.MHz:.3f} MHz")
            else:
                print("Ramsey fit failed; using IF_freq as bare_frequency")
                bare_frequency = meta_dict["IF_freq"]
        self.bare_frequency = bare_frequency

        # ── 1. Reference RB ──────────────────────────────────────────────────
        print("=" * 60)
        print(f"Running reference RB on {element}")
        print("=" * 60)
        ref = RB1Q(
            element,
            circuit_depths,
            num_circuits,
            shots,
            calib_params,
            outer_folder=outer_folder,
            path=path,
            suffix=suffix + "_ref",
            meta_dict=meta_dict,
            config=config,
            bare_frequency=bare_frequency,
            save_h5=False,  # fully captured in the "reference" group of the combined IRB h5 below
            **kwargs,
        )
        p_ref = (ref.data["fit_popt"][1] if ref.data.get("fit_popt") is not None else None)
        p_ref_err = (
            float(np.sqrt(ref.data["fit_pcov"][1][1]))
            if ref.data.get("fit_pcov") is not None
            else np.nan
        )
        self.data["ref"] = {
            k: v for k, v in ref.data.items() if k not in ("meta_dict", "config")
        }

        # ── 2. IRB for each gate ─────────────────────────────────────────────
        gate_fidelities = {}
        gate_fidelity_errors = {}
        for gate in self.gates:
            print("=" * 60)
            print(f"Running IRB: {gate} on {element}")
            print("=" * 60)
            seed = rd.randint(1, 1_000_000)
            prog = self._make_irb_prog(gate, seed)
            job, qm = self.acquire(prog, config, return_qm=True)
            return_prob, fit_d = self._analyze_irb(gate, job, qm, seed)
            self.data[f"irb_{gate}"] = {"return_prob": return_prob, **fit_d}

            p_gate = (fit_d["fit_popt"][1] if fit_d.get("fit_popt") is not None else None)
            p_gate_err = (
                float(np.sqrt(fit_d["fit_pcov"][1][1]))
                if fit_d.get("fit_pcov") is not None
                else np.nan
            )
            if p_ref is not None and p_gate is not None:
                r = (_D - 1) / _D * (1 - p_gate / p_ref)
                r_err = ((_D - 1) / _D) * float(np.sqrt(
                    (p_gate_err / p_ref) ** 2 + (p_gate * p_ref_err / p_ref**2) ** 2
                ))
                gate_fidelities[gate] = float(1 - r)
                gate_fidelity_errors[gate] = r_err
                print(f"  Gate fidelity ({gate}): {1 - r:.6f} ± {r_err:.6f}")

        self.data["gate_fidelities"] = gate_fidelities
        self.data["gate_fidelity_errors"] = gate_fidelity_errors

        # ── 3. Summary table ─────────────────────────────────────────────────
        self._print_summary_table(ref, gate_fidelities, gate_fidelity_errors)

        # ── 4. Summary plot ──────────────────────────────────────────────────
        self._plot_summary(ref, gate_fidelities, gate_fidelity_errors)

        self.data.update({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self._save_irb_h5(ref, gate_fidelities, gate_fidelity_errors)

    # ── HDF5 saving ──────────────────────────────────────────────────────────

    # Maps a gate key to its HDF5 group name; anything not listed uses its own key.
    _GROUP_NAME = {"I": "Identity"}

    # Decay curve is fitRB(x, a, p, b) = a * p**x + b — this order matches fit_popt /
    # fit_popt_err / fit_pcov's rows+columns in every experiment group.
    _FIT_PARAM_NAMES = np.array(["a", "p", "b"], dtype="S")
    _FIT_FUNCTION = "a * p**x + b  (x = num Cliffords); p is the RB decay parameter"

    def _save_irb_h5(self, ref, gate_fidelities, gate_fidelity_errors):
        """
        Writes the full IRB suite as one self-describing HDF5 file:

          /circuit_depths_reference     - depths used by the reference RB
          /circuit_depths_interleaved   - depths used by every interleaved gate (shared)
          /num_circuits, /shots         - shared across reference + every interleaved gate
          /reference/                   - fit_popt, fit_popt_err, fit_pcov, seed,
          /X90/, /mX90/, /Y90/, /mY90/    return_prob, raw_data
          /Identity/                      (fit_popt/_err/_pcov carry an "labels" = [a,p,b] attr)
          /summary/                     - labels/fidelities/fidelity_errors, aligned arrays
                                           covering reference + every interleaved gate

        (h5_data(), the base-class saver, skips every dict-valued self.data entry, which
        is where all of the above actually lives — so it never wrote any of this out.)
        """
        with h5py.File(self.fname, "w") as f:
            f.create_dataset("circuit_depths_reference", data=np.asarray(self.circuit_depths))
            f.create_dataset("circuit_depths_interleaved", data=np.asarray(self.irb_circuit_depths))
            f.create_dataset("num_circuits", data=self.num_circuits)
            f.create_dataset("shots", data=self.shots)

            self._write_experiment_group(f, "reference", ref.data)
            for gate in self.gates:
                group_name = self._GROUP_NAME.get(gate, gate)
                self._write_experiment_group(f, group_name, self.data.get(f"irb_{gate}", {}))

            summary = f.create_group("summary")
            labels = ["reference"] + [self._GROUP_NAME.get(g, g) for g in self.gates]
            fidelities = [ref.data.get("gate_fidelity", np.nan)] + [
                gate_fidelities.get(g, np.nan) for g in self.gates
            ]
            fidelity_errors = [ref.data.get("gate_fidelity_error", np.nan)] + [
                gate_fidelity_errors.get(g, np.nan) for g in self.gates
            ]
            labels_arr = np.array(labels, dtype="S")
            summary.create_dataset("labels", data=labels_arr)
            fid_ds = summary.create_dataset("fidelities", data=np.array(fidelities, dtype=float))
            fid_ds.attrs["labels"] = labels_arr
            err_ds = summary.create_dataset("fidelity_errors", data=np.array(fidelity_errors, dtype=float))
            err_ds.attrs["labels"] = labels_arr

        self._print_h5_contents_report()

    @classmethod
    def _write_experiment_group(cls, f, name, data):
        """One subgroup per experiment (reference or interleaved gate): fit parameters
        and their errors, the RB seed, the return probabilities, and the raw (pre-circuit-
        averaged) shot data, when available. Everything is written as a dataset (not an
        HDF5 attribute) so a plain `.keys()` listing shows all of it; fit_popt/fit_popt_err/
        fit_pcov additionally carry a "labels" = [a, p, b] attribute identifying each entry."""
        grp = f.create_group(name)
        fit_popt = data.get("fit_popt")
        fit_pcov = data.get("fit_pcov")
        if fit_popt is not None:
            popt_ds = grp.create_dataset("fit_popt", data=np.asarray(fit_popt))
            popt_ds.attrs["labels"] = cls._FIT_PARAM_NAMES
            popt_ds.attrs["fit_function"] = cls._FIT_FUNCTION
            if fit_pcov is not None:
                err_ds = grp.create_dataset("fit_popt_err", data=np.sqrt(np.diag(np.asarray(fit_pcov))))
                err_ds.attrs["labels"] = cls._FIT_PARAM_NAMES
        if fit_pcov is not None:
            pcov_ds = grp.create_dataset("fit_pcov", data=np.asarray(fit_pcov))
            pcov_ds.attrs["row_labels"] = cls._FIT_PARAM_NAMES
            pcov_ds.attrs["col_labels"] = cls._FIT_PARAM_NAMES
        if data.get("seed") is not None:
            grp.create_dataset("seed", data=data["seed"])
        if data.get("return_prob") is not None:
            grp.create_dataset("return_prob", data=np.asarray(data["return_prob"]))
        if data.get("raw_data") is not None:
            grp.create_dataset("raw_data", data=np.asarray(data["raw_data"]))

    def _print_h5_contents_report(self):
        """Print exactly what landed in each group, so a fit/acquisition failure that leaves
        a group sparse (as opposed to a saving bug) is obvious immediately from the console."""
        with h5py.File(self.fname, "r") as f:
            print(f"\n  IRB HDF5 saved to {self.fname}")
            print(f"    root: {sorted(f.keys())}")
            for name in ["reference"] + [self._GROUP_NAME.get(g, g) for g in self.gates] + ["summary"]:
                if name in f:
                    print(f"    {name}/: {sorted(f[name].keys())}")
                else:
                    print(f"    {name}/: MISSING GROUP")

    # ── Summary table ────────────────────────────────────────────────────────

    def _print_summary_table(self, ref, gate_fidelities, gate_fidelity_errors):
        W = 60
        print("\n" + "=" * W)
        print(f"  IRB RESULTS — {self.element}")
        print("=" * W)

        # Reference RB row
        ref_fid = ref.data.get("gate_fidelity")
        ref_err = ref.data.get("gate_fidelity_error")
        if ref_fid is not None:
            ref_str = f"{ref_fid:.6f} ± {ref_err:.6f}" if ref_err is not None else f"{ref_fid:.6f}"
        else:
            ref_str = "fit failed"
        print(f"  {'Reference RB':<12}  {ref_str}")
        print("-" * W)

        # Gate rows
        print(f"  {'Gate':<8}  {'Fidelity':>14}   {'Error (1σ)':>12}")
        print("-" * W)
        for gate in self.gates:
            fid = gate_fidelities.get(gate)
            err = gate_fidelity_errors.get(gate)
            if fid is not None:
                fid_str = f"{fid:.6f}"
                err_str = f"± {err:.6f}" if err is not None else ""
            else:
                fid_str = "fit failed"
                err_str = ""
            print(f"  {gate:<8}  {fid_str:>14}   {err_str:>12}")

        print("=" * W + "\n")

    # ── QUA program ───────────────────────────────────────────────────────────

    def _make_irb_prog(self, gate, seed):
        """Build the QUA program for one interleaved gate.

        For circuit depth m (m random Cliffords), the sequence array layout is:
          [C1, G, C2, G, ..., Cm, G, Recovery]  at indices [0..2m].
        We play indices 0 through 2m inclusive (depth argument = 2m).
        """
        gate_idx = get_cayley_idx(gate)
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        ss_thresh = self.calib_params["threshold"]
        ground_thresh = self.calib_params["ground_threshold"]
        identity_phase_shift = (
            (self.meta_dict["IF_freq"] - self.bare_frequency) / u.GHz * 16
            if self.bare_frequency is not None else 0.0
        )

        with program() as prog:
            k = declare(int)
            m = declare(int)
            m2 = declare(int)   # 2*m: actual array index for depth m
            n = declare(int)
            state = declare(int)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence_irb(
                    seed, self.irb_max_depth, gate_idx
                )
                with for_each_(m, self.irb_circuit_depths):
                    assign(m2, 2 * m)
                    assign(saved_gate, sequence_list[m2])
                    assign(sequence_list[m2], inv_gate_list[m2 - 1])

                    with for_(n, 0, n < self.shots, n + 1):
                        wait(resonator_dump_time * u.ns, self.element)
                        reset_frame(qubit)
                        align()
                        play_sequence(qubit, sequence_list, m2)

                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_stream)
                        align(qubit, resonator)

                        with while_(I_reset > ground_thresh):
                            with if_(I_reset > ss_thresh):
                                play("X180", qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)

                    assign(sequence_list[m2], saved_gate)
                save(k, k_stream)

            with stream_processing():
                k_stream.save("iteration")
                ss_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(
                    len(self.irb_circuit_depths)
                ).save_all("ss")

        return prog

    # ── Live analysis ─────────────────────────────────────────────────────────

    def _analyze_irb(self, gate, job, qm, seed):
        """Fetch live stream, fit, plot, and close qm. Returns (return_prob, fit_data)."""
        results = fetching_tool(job, data_list=["ss", "iteration"], mode="live")

        def fitRB(x, a, p, b):
            return a * p ** x + b

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        (line,) = ax.plot([], [], "o", ms=4, label="data")
        (fit_line,) = ax.plot([], [], "--", color="C0", alpha=0.7, label="fit")
        ax.set_xlabel("Num Cliffords")
        ax.set_ylabel("Return probability")
        ax.set_xscale("log")
        ax.set_xlim(self.irb_circuit_depths[0], self.irb_circuit_depths[-1])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.suptitle(f"IRB: {gate} on {self.element}")
        interrupt_on_close(fig, job)

        ss_all = None
        return_prob = None
        while results.is_processing():
            ss_all, iteration = results.fetch_all()
            if ss_all.dtype.names:
                ss_all = ss_all["value"]
            if ss_all.shape[0] == 0:
                continue
            print(f"  Circuit: {iteration}/{self.num_circuits}", end="\r", flush=True)
            return_prob = 1 - np.mean(ss_all, axis=0)
            line.set_data(self.irb_circuit_depths, return_prob)
            try:
                popt, pcov = curve_fit(
                    fitRB, self.irb_circuit_depths, return_prob,
                    [0.4, 0.99, 0.5], maxfev=5000,
                )
                p, p_err = popt[1], np.sqrt(pcov[1][1])
                fit_x = np.linspace(self.irb_circuit_depths[0], self.irb_circuit_depths[-1], 200)
                fit_line.set_data(fit_x, fitRB(fit_x, *popt))
                ax.set_title(f"p = {p:.6f} ± {p_err:.6f}")
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view(scaley=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        fit_data = {"seed": seed}
        if ss_all is not None:
            # Per-circuit, per-depth shot-averaged excited fraction, before averaging over
            # circuits into return_prob. Kept as the closest available "raw data".
            fit_data["raw_data"] = ss_all
        if return_prob is not None:
            try:
                popt, pcov = curve_fit(
                    fitRB, self.irb_circuit_depths, return_prob,
                    [0.4, 0.99, 0.5], maxfev=5000,
                )
                p, p_err = popt[1], np.sqrt(pcov[1][1])
                print(f"  {gate}: p = {p:.6f} ± {p_err:.6f}")
                fit_data.update({"fit_popt": popt, "fit_pcov": pcov})
            except Exception:
                print(f"  {gate}: fit failed.")

        gate_iname = self.iname.replace(".jpg", f"_{gate}.jpg")
        fig.savefig(gate_iname, bbox_inches="tight")
        plt.close(fig)
        qm.close()

        return return_prob, fit_data

    # ── Summary plot ──────────────────────────────────────────────────────────

    def _plot_summary(self, ref, gate_fidelities, gate_fidelity_errors):
        def fitRB(x, a, p, b):
            return a * p ** x + b

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        x_max = max(self.circuit_depths[-1], self.irb_circuit_depths[-1])
        x_min = min(self.circuit_depths[0], self.irb_circuit_depths[0])
        x_ref_fit = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 300)
        x_irb_fit = np.linspace(self.irb_circuit_depths[0], self.irb_circuit_depths[-1], 300)

        ref_prob = ref.data.get("return_prob")
        if ref_prob is not None:
            ref_label = "Ref RB"
            ref_fid = ref.data.get("gate_fidelity")
            ref_err = ref.data.get("gate_fidelity_error")
            if ref_fid is not None:
                ref_label += f"  F={ref_fid:.5f}"
                if ref_err is not None:
                    ref_label += f"±{ref_err:.1e}"
            ax.plot(self.circuit_depths, ref_prob, "o", color="black", ms=4, label=ref_label)
        if ref.data.get("fit_popt") is not None:
            ax.plot(x_ref_fit, fitRB(x_ref_fit, *ref.data["fit_popt"]), "--", color="black", alpha=0.6)

        colors = plt.cm.tab10(np.linspace(0, 0.9, len(self.gates)))
        for gate, color in zip(self.gates, colors):
            irb_d = self.data.get(f"irb_{gate}", {})
            rp = irb_d.get("return_prob")
            fid = gate_fidelities.get(gate)
            err = gate_fidelity_errors.get(gate)
            label = gate
            if fid is not None:
                label += f"  F={fid:.5f}"
                if err is not None:
                    label += f"±{err:.1e}"
            if rp is not None:
                ax.plot(self.irb_circuit_depths, rp, "o", color=color, ms=4, label=label)
            popt = irb_d.get("fit_popt")
            if popt is not None:
                ax.plot(x_irb_fit, fitRB(x_irb_fit, *popt), "--", color=color, alpha=0.6)

        ax.set_xlabel("Num Cliffords")
        ax.set_ylabel("Return probability")
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.4, 1)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.suptitle(f"IRB Summary: {self.element}")

        summary_iname = self.iname.replace(".jpg", "_summary.jpg")
        fig.savefig(summary_iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()


# ── Sequence generator for IRB ────────────────────────────────────────────────

def generate_sequence_irb(seed, longest_circuit, interleaved_gate_idx):
    """
    Generate an interleaved RB sequence of length 2*longest_circuit + 1.

    Array layout for depth m (m random Cliffords):
      index  0     1     2     3     ...  2m-2   2m-1   2m
      gate   C1    G     C2    G     ...  Cm     G      Recovery

    Recovery = inv_gate[2m-1], placed at sequence[2m] by make_prog.
    """
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=2 * longest_circuit + 1)
    inv_gate = declare(int, size=2 * longest_circuit + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < 2 * longest_circuit, i + 2):
        # Random Clifford
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])
        # Interleaved gate
        assign(step, interleaved_gate_idx)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    return sequence, inv_gate


def get_cayley_idx(gate):
    mapping = {
        "I":    0,
        "X180": 1,
        "Y180": 2,
        "X90":  12,
        "mX90": 13,
        "Y90":  14,
        "mY90": 15,
    }
    if gate not in mapping:
        raise ValueError(f"Unknown gate '{gate}'. Valid: {list(mapping)}")
    return mapping[gate]


if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q
    from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q

    QUBIT = "q4"
    CIRCUIT_DEPTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 7000]
    NUM_CIRCUITS = 50
    SHOTS = 50
    SS_SHOTS = 1000

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    outer_folder = meta_dict["global_params"]["outer_folder"]

    # Single-shot calibration
    ss_cal = SingleShot1Q(
        QUBIT,
        SS_SHOTS,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=QUBIT,
        suffix="SS_cal",
        config=config,
    )
    calib_params = ss_cal.calib_params
    meta_dict["read_theta"] = ss_cal.calib_params["read_theta"]
    config = meta_dict_to_config_1Q(meta_dict)  # rebuild with updated read_theta

    exp = RB1Q_Interleaved(
        QUBIT,
        CIRCUIT_DEPTHS,
        NUM_CIRCUITS,
        SHOTS,
        calib_params,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=QUBIT,
        suffix="IRB_suite",
        config=config,
    )
