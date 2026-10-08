from qm.qua import *
from datetime import datetime
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential_2, exponential_2_forced_origin, T2CPMG_seed
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qm import simulate
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class T2CPMG(ExperimentClass):
    """
    T2CPMG experiment
    """
    def __init__(self, element, t_vec, num_pi, shots, calib_params, outer_folder='', path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element
        self.t_vec = t_vec
        self.tau_vec_normalized_clk = np.array([int(t / (8 * num_pi)) for t in t_vec], dtype=int)
        self.shots = shots
        self.calib_params = calib_params
        self.num_pi = num_pi

        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec,
            'shots': self.shots,
            'num_pi': self.num_pi,
            'meta_dict': self.meta_dict
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        shots = self.shots
        reset_time = self.meta_dict
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            n_pi = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(t, self.tau_vec_normalized_clk):
                    play('X90', qubit)
                    wait(t, qubit)
                    # Qua does not like division. adds 400 ns latency
                    with for_(n_pi, 0, n_pi < self.num_pi - 1, n_pi + 1):
                        play('Y180', qubit)
                        wait(2 * t, qubit)
                    play('Y180', qubit)
                    wait(t, qubit)
                    play('mX90', qubit)

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
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.tau_vec_normalized_clk)).save_all("ss")
            return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")
        t_vec = self.t_vec / u.us

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line, = ax.plot([], [], 'o', label='data')
        fit_line, = ax.plot([], [], '-', alpha=0.7, label='fit')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Excited state probability')
        ax.set_xlim(t_vec[0], t_vec[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f"{self.element}, T2CPMG, {self.num_pi} pulses")
        interrupt_on_close(fig, job)

        ss_all = None
        while results.is_processing():
            ss_all = results.fetch_all()[0]
            if ss_all.shape[0] == 0:
                continue
            ss_array = np.mean(ss_all, axis=0)
            print(f"Shot: {ss_all.shape[0]}/{self.shots}", end="\r", flush=True)
            line.set_data(t_vec, ss_array)
            try:
                seed = T2CPMG_seed(self.t_vec, ss_array)
                bounds = [[0, 0, 0, 0], [np.inf, 1, np.inf, 1]]
                popt, pcov = curve_fit(exponential_2, t_vec, ss_array, p0=seed, bounds=bounds)
                T2, T2_err = popt[2], np.sqrt(pcov[2][2])
                fit_line.set_data(t_vec, exponential_2(t_vec, *popt))
                ax.set_title(f"T2CPMG = {T2:.2f} ± {T2_err:.2f} us")
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        ss_array = np.mean(ss_all, axis=0) if ss_all is not None else np.zeros(len(t_vec))
        self.data.update({'ss_data': ss_array})

        try:
            seed = T2CPMG_seed(self.t_vec, ss_array)
            bounds = [[0, 0, 0, 0], [np.inf, 1, np.inf, 1]]
            popt, pcov = curve_fit(exponential_2, t_vec, ss_array, p0=seed, bounds=bounds)
            T2 = popt[2]
            n = popt[0]
            T2_err = np.sqrt(pcov[2][2])
            n_err = np.sqrt(pcov[0][0])
            self.data.update({'T2CPMG': (T2, T2_err), 'fit_popt': popt, 'fit_pcov': pcov})
            ax.set_title(f"{self.element}, {self.num_pi} pulses, T2CPMG = {T2:.2f} ± {T2_err:.2f} us, n = {n:.4f} ± {n_err:.4f}")
            print(f"T2CPMG = {T2:.2f} ± {T2_err:.2f} us")
        except Exception:
            print("T2CPMG Fit Failed")
            ax.set_title(f"{self.element}, {self.num_pi} pulses, T2CPMG")

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()
