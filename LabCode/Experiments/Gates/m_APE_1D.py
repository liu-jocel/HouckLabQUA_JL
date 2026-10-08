from qm.qua import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib

matplotlib.use("qtagg", force=True)
from datetime import datetime
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import cosine, T2R_seed
from qualang_tools.results import progress_counter, fetching_tool
from LabCode.Helpers.pulse_macros import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class APE1D(ExperimentClass):
    """
    T2R experiement
    """
    def __init__(self,element, phi_vec, n_vec, shots, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.n_vec = n_vec
        self.shots = shots
        self.phi_vec = phi_vec
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "n_vec": self.n_vec,
            'shots': self.shots,
            'phi_vec': self.phi_vec,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            m = declare(int)
            m_max = declare(int)
            p = declare(fixed)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(m_max, self.n_vec):
                    with for_each_(p, self.phi_vec):
                        reset_frame(qubit)
                        play("X90", qubit)
                        with for_(m, 0, m < m_max, m + 1):
                            play("X90", qubit)
                            wait(self.meta_dict['X90']['length']*u.ns, qubit)
                            play("mX90", qubit)
                        frame_rotation_2pi(p, qubit)
                        play("X90", qubit)

                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_stream)
                        align(qubit, resonator)

                        # Active reset
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play_pulse(qubit, "X180",
                                           phase=self.meta_dict["X180"]['phase'])
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time*u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.phi_vec)).buffer(len(self.n_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):

        result = fetching_tool(self.job, data_list=["ss"], mode='live')

        # --- Figure setup ---
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"{self.element} Ramsey Phase Calibration")

        ax.set_xlabel("Phase / 2π")
        ax.set_ylabel("Excited state probability")
        ax.set_ylim(0, 1)
        ax.grid(True)

        interrupt_on_close(fig, job)

        # --- Colormap ---
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(self.n_vec)))

        # --- Plot objects ---
        data_lines = []
        fit_lines = []

        for c, n in zip(colors, self.n_vec):
            (ldata,) = ax.plot(self.phi_vec, np.zeros_like(self.phi_vec), "o",
                               color=c, ms=4, label=f"n={n}")
            (lfit,) = ax.plot(self.phi_vec, np.zeros_like(self.phi_vec), "-",
                              color=c, lw=1.5)
            data_lines.append(ldata)
            fit_lines.append(lfit)

        ax.legend(loc="best", ncol=2, fontsize=9)

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        # ---------- Storage ----------
        phi0_vec = np.zeros(len(self.n_vec))
        amp_vec = np.zeros(len(self.n_vec))
        offset_vec = np.zeros(len(self.n_vec))

        # ---------- Fit model ----------
        def cosine_fixed_period(x, A, phi0, C):
            """
            Cosine with fixed period = 1:
            y = A cos(2π x + phi0) + C
            """
            return A * np.cos(2 * np.pi * x + phi0) + C

        # ---------- Live loop ----------
        while result.is_processing():

            ss = result.fetch_all()[0]  # shape: (len(n_vec), len(phi_vec))

            if ss.size == 0:
                plt.pause(0.05)
                continue

            self.data["populations"] = ss

            # --- Fit cosines per n ---
            for j, n in enumerate(self.n_vec):

                y = ss[j, :]

                # Skip if not yet populated
                if np.all(y == 0):
                    continue

                # Initial guesses
                A0 = 0.5 * (np.max(y) - np.min(y))
                C0 = np.mean(y)
                phi0_guess = 0.0

                try:
                    popt, _ = curve_fit(
                        cosine_fixed_period,
                        self.phi_vec,
                        y,
                        p0=[A0, phi0_guess, C0],
                        maxfev=5000
                    )

                    A, phi0, C = popt

                    amp_vec[j] = A
                    phi0_vec[j] = phi0
                    offset_vec[j] = C

                    # Plot fit
                    phi_fit = np.linspace(self.phi_vec[0], self.phi_vec[-1], 400)
                    fit_lines[j].set_data(phi_fit, cosine_fixed_period(phi_fit, *popt))

                except RuntimeError:
                    continue

                # Plot raw data
                data_lines[j].set_data(self.phi_vec, y)

            plt.pause(0.05)
            plt.pause(0.03)

        # ---------- Final results ----------
        phi0_vec = np.unwrap(phi0_vec)

        self.data.update({
            "phi0_vec": phi0_vec,
            "amp_vec": amp_vec,
            "offset_vec": offset_vec,
            "n_vec": self.n_vec
        })

        print("\nFinal fitted phase offsets:")
        for n, phi in zip(self.n_vec, phi0_vec):
            print(f"  n = {n:3d} :  φ0 / 2π = {phi / (2 * np.pi): .5f}")

        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)


