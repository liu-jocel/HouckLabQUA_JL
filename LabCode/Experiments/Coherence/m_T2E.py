from qm.qua import *
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from datetime import datetime

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
import warnings
from scipy.optimize import OptimizeWarning
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential, T2E_seed
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


class T2E(ExperimentClass):
    """
    T2E experiment
    """

    def __init__(
        self,
        element,
        t_vec,
        shots,
        calib_params,
        outer_folder="",
        path="T2R",
        suffix="data",
        meta_dict=None,
        config=None,
    ):
        super().__init__(
            path=path,
            outer_folder=outer_folder,
            suffix=suffix,
            meta_dict=meta_dict,
            confg=config,
        )
        self.config = config
        self.element = element
        self.t_vec = t_vec * u.ns
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "shots": self.shots,
            "meta_dict": self.meta_dict,
            "num_pi": 1,
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        with program() as prog:
            n = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec / 2 * u.ns)):
                    play("X90", qubit)
                    wait(t, qubit)
                    play("Y90", qubit)
                    play("Y90", qubit)
                    wait(t, qubit)
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
                            play("X180", qubit)
                            align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).save_all("ss")
        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")
        t_vec = self.t_vec / u.us

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line, = ax.plot([], [], "o", label="data")
        fit_line, = ax.plot([], [], "-", alpha=0.7, label="fit")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Excited state probability")
        ax.set_xscale("log")
        ax.set_xlim(t_vec[0], t_vec[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f"{self.element}, T2E")
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
                seed = T2E_seed(t_vec, ss_array)
                popt, pcov = curve_fit(exponential, t_vec, ss_array, p0=seed, maxfev=5000)
                T2 = popt[2]
                T2_err = np.sqrt(pcov[2][2])
                fit_line.set_data(t_vec, exponential(t_vec, *popt))
                ax.set_title(f"T2E = {T2:.4f} ± {T2_err:.4f} us")
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        ss_array = np.mean(ss_all, axis=0) if ss_all is not None else np.zeros(len(t_vec))
        self.data.update({"ss_data": ss_array})

        warnings.simplefilter("error", OptimizeWarning)
        try:
            seed = T2E_seed(t_vec, ss_array)
            popt, pcov = curve_fit(exponential, t_vec, ss_array, p0=seed, maxfev=5000)
            T2 = popt[2]
            T2_err = np.sqrt(pcov[2][2])
            self.data.update({"T2E": (T2, T2_err), "fit": (popt, pcov)})
            ax.set_title(f"{self.element}, T2E = {T2:.4f} ± {T2_err:.4f} us")
            print(f"T2E = {T2:.4f} ± {T2_err:.4f} us")
        except Exception:
            print("T2E Fit Failed")
            ax.set_title(f"{self.element}, T2E")

        self.data["t_vec"] = t_vec
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()
