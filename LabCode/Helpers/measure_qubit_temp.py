# Add repo root to path so this file can be run directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

"""
Measure qubit temperature from thermal population distribution.

Procedure:
1. Runs SingleShot1Q3Level calibration internally to get discrimination parameters
2. Runs readout-only shots (no qubit prep) to measure thermal populations
3. Classifies each shot into |0>, |1>, |2> using the SS3 discriminators
4. Fits populations to a 3-level Boltzmann distribution
5. Prints and saves effective qubit temperature

Usage:
    from LabCode.Helpers.measure_qubit_temp import QubitTemp
    qt = QubitTemp(QUBIT, shots=2000, ss3_shots=600, meta_dict=meta_dict,
                   outer_folder=meta_dict["global_params"]["outer_folder"],
                   config=config)
    print(qt.data["T_eff_boltzmann_mK"], "mK")
"""

from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from scipy.constants import k as k_B, hbar
from scipy.optimize import minimize_scalar
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


class QubitTemp(ExperimentClass):
    """
    Measure effective qubit temperature from thermal populations.

    Runs its own 3-level SS calibration, then measures thermal state without
    any qubit preparation. Saves data under:
        <outer_folder>/<qubit>/<qubit>_<YYYY_MM_DD>/<qubit>_<HH_MM_SS>_<suffix>.h5
    """

    def __init__(
        self,
        element,
        shots=2000,
        ss3_shots=600,
        outer_folder="",
        path=None,
        suffix="data",
        meta_dict=None,
        config=None,
        ss3_params_path=None,
        do_3L_cal=True,
        **kwargs,
    ):
        """
        do_3L_cal : bool
            True  → run SS3Cal and save params to ss3_params_path.
            False → load params from ss3_params_path; raises a clear error if
                    the file is not found (no hardware call is made).
        ss3_params_path : str or Path, optional
            Path to the JSON cache file. Defaults to
            outer_folder/<element>_ss3_params.json.
        """
        import json, os

        # Use qubit name as path so folder is  outer_folder/q4/q4_<date>/
        if path is None:
            path = element

        # Default cache location: outer_folder/<element>_ss3_params.json
        if ss3_params_path is None and outer_folder:
            ss3_params_path = os.path.join(outer_folder, f"{element}_ss3_params.json")

        super().__init__(
            path=path,
            outer_folder=outer_folder,
            suffix=suffix,
            meta_dict=meta_dict,
            config=config,
            **kwargs,
        )

        self.element   = element
        self.shots     = shots
        self.ss3_shots = ss3_shots

        self.data = {
            "qubit":     self.element,
            "shots":     self.shots,
            "ss3_shots": self.ss3_shots,
            "meta_dict": self.meta_dict,  # skipped by h5_data (dict)
        }

        # Step 1: SS3Cal — run fresh cal or load from cache
        if do_3L_cal:
            print(f"\n--- {element}: Running 3-level SS calibration ({ss3_shots} shots) ---")
            self.ss3 = SingleShot1Q3Level(
                element,
                ss3_shots,
                outer_folder=outer_folder,
                path=path,
                suffix=f"{suffix}_SS3Cal",
                meta_dict=meta_dict,
                config=self.config,
                flat_path=kwargs.get("flat_path", False),
            )
            print(f"    SS3 max fidelity: {self.ss3.max_F:.3f}")
            self._load_discriminators_from_ss3()

            if ss3_params_path is not None:
                params = {
                    "theta_01":  self.theta_01,
                    "theta_12":  self.theta_12,
                    "thresh_01": self.thresh_01,
                    "thresh_12": self.thresh_12,
                    "max_F":     float(self.ss3.max_F),
                }
                os.makedirs(os.path.dirname(os.path.abspath(ss3_params_path)), exist_ok=True)
                with open(ss3_params_path, "w") as f:
                    json.dump(params, f, indent=2)
                print(f"    SS3 params saved to {ss3_params_path}")
        else:
            if ss3_params_path is None or not os.path.isfile(ss3_params_path):
                raise FileNotFoundError(
                    f"do_3L_cal=False but SS3 params file not found: {ss3_params_path}\n"
                    f"Run once with do_3L_cal=True to generate it."
                )
            print(f"\n--- {element}: Loading SS3 params from {ss3_params_path} ---")
            with open(ss3_params_path, "r") as f:
                p = json.load(f)
            self.theta_01  = p["theta_01"]
            self.theta_12  = p["theta_12"]
            self.thresh_01 = p["thresh_01"]
            self.thresh_12 = p["thresh_12"]
            print(f"    theta_01={self.theta_01:.4f}  thresh_01={self.thresh_01:.6e}")
            print(f"    theta_12={self.theta_12:.4f}  thresh_12={self.thresh_12:.6e}")
            print(f"    SS3 max fidelity (cached): {p.get('max_F', float('nan')):.3f}")

        # Step 2: thermal population measurement
        print(f"\n--- {element}: Measuring thermal populations ({shots} shots, no prep) ---")
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.h5_data()

    # ------------------------------------------------------------------
    def _load_discriminators_from_ss3(self):
        disc_01 = self.ss3.data.get("01_discrim", {})
        disc_12 = self.ss3.data.get("12_discrim", {})
        self.theta_01  = disc_01.get("Theta",     0.0)
        self.theta_12  = disc_12.get("Theta",     0.0)
        self.thresh_01 = disc_01.get("Threshold", 0.0)
        self.thresh_12 = disc_12.get("Threshold", 0.0)

    # ------------------------------------------------------------------
    def make_prog(self):
        shots = self.shots
        with program() as prog:
            n        = declare(int)
            I        = declare(fixed)
            Q        = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                # No qubit prep — measure thermal state directly
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream)
                save(Q, Q_stream)
            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
        return prog

    # ------------------------------------------------------------------
    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["I", "Q"], mode="live")

        I_data = None
        Q_data = None
        while results.is_processing():
            I_data, Q_data = results.fetch_all()
            if I_data is not None and I_data.dtype.names:
                I_data = I_data["value"]
                Q_data = Q_data["value"]

        qm.close()

        # --- classify each shot into |0>, |1>, |2> ---
        # 01 discriminator: rotated so |1> sits at higher real values than |0>.
        #   real < thresh_01 → |0>
        #   real > thresh_01 → |1> or |2>   (use 12 disc to resolve)
        # 12 discriminator: rotated so |2> sits at higher real values than |1>.
        #   real < thresh_12 → |1>
        #   real > thresh_12 → |2>
        c        = I_data + 1j * Q_data
        c_rot_01 = np.exp(-1j * self.theta_01) * c
        c_rot_12 = np.exp(-1j * self.theta_12) * c

        is_not_gnd = np.real(c_rot_01) > self.thresh_01   # |1> or |2>
        is_2nd_exc = np.real(c_rot_12) > self.thresh_12   # |2>

        state = np.zeros(len(c), dtype=int)
        state[is_not_gnd & ~is_2nd_exc] = 1   # right of 01, left of 12 → |1>
        state[is_not_gnd &  is_2nd_exc] = 2   # right of 01, right of 12 → |2>

        n_0   = int(np.sum(state == 0))
        n_1   = int(np.sum(state == 1))
        n_2   = int(np.sum(state == 2))
        total = n_0 + n_1 + n_2

        pop_0 = n_0 / total
        pop_1 = n_1 / total
        pop_2 = n_2 / total

        print(f"  |0>: {pop_0:.4f}  ({n_0}/{total} = {100*pop_0:.1f}%)")
        print(f"  |1>: {pop_1:.4f}  ({n_1}/{total} = {100*pop_1:.1f}%)")
        print(f"  |2>: {pop_2:.4f}  ({n_2}/{total} = {100*pop_2:.1f}%)")

        # --- qubit energies ---
        # ω_01 = 2π(LO + IF),  ω_12 = ω_01 + α  (α = anharmonicity, signed)
        # E_0 = 0,  E_1 = ℏω_01,  E_2 = E_1 + ℏω_12 = ℏ(2ω_01 + α)
        # In units of ℏω_01: E = [0, 1, 2 + α/ω_01]
        qubit_freq_Hz = self.meta_dict["q_LO"]["LO_freq"] + self.meta_dict["IF_freq"]
        anharm_Hz     = self.meta_dict["anharmonicity"]   # stored in Hz
        anharm_norm   = anharm_Hz / qubit_freq_Hz         # α / ω_01 (dimensionless)

        # --- Boltzmann fit: minimise residual over β = ℏω_01/(kT) ---
        def boltzmann_pops(beta):
            E       = np.array([0.0, 1.0, 2.0 + anharm_norm])
            weights = np.exp(-beta * E)
            return weights / weights.sum()

        def residual(beta):
            p = boltzmann_pops(beta)
            return (p[0]-pop_0)**2 + (p[1]-pop_1)**2 + (p[2]-pop_2)**2

        fit      = minimize_scalar(residual, bounds=(0.1, 100.0), method="bounded")
        beta_fit = fit.x   # ℏω_01 / kT

        # T = ℏω_01 / (k_B · β) = h·f / (k_B · β)
        k_B_per_h = k_B / (2.0 * np.pi * hbar)   # Hz / K  (= k_B / h)
        T_eff     = qubit_freq_Hz / (k_B_per_h * beta_fit)

        # Cross-check: simple two-level estimate from P0/P1 ratio
        if pop_0 > 0 and pop_1 > 0:
            T_ratio = qubit_freq_Hz / (k_B_per_h * np.log(pop_0 / pop_1))
        else:
            T_ratio = np.nan

        # f_02 = f_01 + f_12 = f_01 + (f_01 + α) = 2·f_01 + α
        qubit_freq_02_Hz = 2 * qubit_freq_Hz + anharm_Hz

        print(f"\n  Qubit f_01 (LO + IF)          : {qubit_freq_Hz / u.GHz:.6f} GHz")
        print(f"  Qubit f_02 (2·f_01 + α)       : {qubit_freq_02_Hz / u.GHz:.6f} GHz")
        print(f"  Anharmonicity                 : {anharm_Hz / u.MHz:.3f} MHz")
        fitted_pops = boltzmann_pops(beta_fit)
        #print(f"  Fitted β = ℏω_01/kT           : {beta_fit:.4f}")
        print(f"  T_eff (3-level Boltzmann fit) : {T_eff * 1e3:.1f} mK"
              f"  (P0={fitted_pops[0]:.4f}, P1={fitted_pops[1]:.4f}, P2={fitted_pops[2]:.4f})")
        print(f"  T_eff (ln P0/P1 ratio)        : {T_ratio * 1e3:.1f} mK")

        # --- figure: 2D histogram of raw I,Q data ---
        from matplotlib.colors import LogNorm
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(cmap(0.0))   # zero-count (NaN under LogNorm) → darkest colour
        fig, ax = plt.subplots(figsize=(6, 5))
        _, _, _, img = ax.hist2d(
            I_data, Q_data,
            bins=100,
            cmap=cmap,
            norm=LogNorm(vmin=1),
        )
        fig.colorbar(img, ax=ax, label="Counts (log scale)")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title(
            f"{self.element}  —  Thermal IQ  ({self.shots} shots, no prep)\n"
            f"f₀₁ = {qubit_freq_Hz/u.GHz:.4f} GHz,  α = {anharm_Hz/u.MHz:.0f} MHz\n"
            f"T_eff = {T_eff*1e3:.1f} mK  |  "
            f"P0={pop_0:.3f}  P1={pop_1:.3f}  P2={pop_2:.3f}"
        )
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

        self.data.update({
            "I_data":               I_data,
            "Q_data":               Q_data,
            "state_classification": state,
            "pop_0":                pop_0,
            "pop_1":                pop_1,
            "pop_2":                pop_2,
            "n_0":                  n_0,
            "n_1":                  n_1,
            "n_2":                  n_2,
            "qubit_freq_Hz":        qubit_freq_Hz,
            "anharmonicity_Hz":     anharm_Hz,
            "beta_fit":             beta_fit,
            "T_eff_boltzmann_mK":   T_eff  * 1e3,
            "T_eff_ratio_mK":       T_ratio * 1e3,
            "ss3_max_fidelity":     getattr(getattr(self, "ss3", None), "max_F", float("nan")),
        })


if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    QUBIT        = "q1"
    meta_dict    = make_meta_dict(QUBIT)[QUBIT]
    config       = meta_dict_to_config_1Q(meta_dict)
    outer_folder = meta_dict["global_params"]["outer_folder"]

    # do_3L_cal=True  → run SS3Cal and save params to outer_folder/q4_ss3_params.json
    # do_3L_cal=False → load from that file (error if missing)
    qt = QubitTemp(
        QUBIT,
        shots=6000,
        ss3_shots= 1000,
        outer_folder=outer_folder,
        meta_dict=meta_dict,
        config=config,
        do_3L_cal=True,
    )
