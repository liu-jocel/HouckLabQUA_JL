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

class RamseyPhase(ExperimentClass):
    """
    T2R experiement
    """
    def __init__(self,element, phi_vec, tau, shots, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.tau = tau,
        self.shots = shots
        self.phi_vec = phi_vec
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "tau": self.tau,
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
            p = declare(fixed)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(p, self.phi_vec):
                    reset_frame(qubit)
                    play_pulse(qubit, "X90",
                               phase=0)
                    # wait(4, qubit)
                    frame_rotation_2pi(p, qubit)
                    play_pulse(qubit, "Y90",
                               phase=0)#self.meta_dict["Y90"]['phase'])

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
                    # wait(3 * u.ms, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.phi_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):

        # Extract data
        result = fetching_tool(self.job, data_list=["ss"], mode='live')

        # Live Plotting
        fig, ax = plt.subplots()
        fig.suptitle(f"{self.element} Ramsey Phase Measurement")
        ax.set_xlabel('Phase / 2π')
        ax.set_ylabel('Excited state probability')

        (line,) = ax.plot(self.phi_vec, [0.5] * len(self.phi_vec), "o",
                          color="darkblue", lw=1, label="Data")
        (fit_line,) = ax.plot(self.phi_vec, [0.5] * len(self.phi_vec), "-",
                              color="darkblue", lw=1, label="Linear Fit")

        # Horizontal reference line at 0.5
        hline = ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="0.5 Reference")

        ax.legend(loc="best")

        interrupt_on_close(fig, job)  # Interrupt job when closing the figure

        while result.is_processing():
            populations = result.fetch_all()[0]
            self.data.update({'populations': populations})

            # Plot raw data
            line.set_data(self.phi_vec, populations)

            # ---- Linear fit + intercept extraction ----
            try:
                # Linear model
                def linear(x, m, b):
                    return m * x + b

                # Initial guess
                m_guess = (populations[-1] - populations[0]) / \
                          (self.phi_vec[-1] - self.phi_vec[0])
                b_guess = np.mean(populations)

                popt, pcov = curve_fit(linear, self.phi_vec, populations,
                                       p0=[m_guess, b_guess])

                m, b = popt

                # Solve for intercept at y = 0.5 → 0.5 = m x + b
                if abs(m) > 1e-12:
                    x_intercept = (0.5 - b) / m
                    print(f"\rCrossing phase: {x_intercept:8.5f}", end="", flush=True)
                else:
                    print("Slope ~ 0 → no meaningful intercept.")

                # Plot fit
                phi_fit = np.linspace(self.phi_vec[0], self.phi_vec[-1], 200)
                fit_line.set_data(phi_fit, linear(phi_fit, *popt))

            except Exception as e:
                print("Linear fit failed:", e)
            # Refresh plot
            ax.relim()
            ax.autoscale_view(scaley=True)
            plt.pause(0.05)

            plt.pause(0.01)
        self.data["phase"] = x_intercept
        print(f"\rCrossing phase: {x_intercept:8.5f}")
        print("")
        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)


