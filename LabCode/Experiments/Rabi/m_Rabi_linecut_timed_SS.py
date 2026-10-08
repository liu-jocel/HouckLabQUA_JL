from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from LabCode.Helpers.fit_functions import decaying_cosine, rabi_seed
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from qualang_tools.plot.fitting import Fit

u = unit(coerce_to_integer=True)


class RabiLinecutTimedSS(ExperimentClass):
    """
    Time Rabi Linecut with single shot readout
    """

    def __init__(
        self,
        element,
        t_vec,
        shots,
        calib_params,
        ampx=1.0,
        pulse_name="constant_tone",
        outer_folder="",
        path="Rabi",
        suffix="data",
        meta_dict=None,
        config=None,
    ):
        super().__init__(
            path=path,
            outer_folder=outer_folder,
            suffix=suffix,
            meta_dict=meta_dict,
            config=config,
        )

        self.element = element  # meta dict element name
        self.pulse_name = pulse_name
        self.t_vec = t_vec
        self.ampx = ampx

        self.shots = shots
        self.calib_params = calib_params

        self.read_len = self.meta_dict["global_params"]["read_len"]

        self.data = {
            "qubit": self.element,
            "pulse_name": self.pulse_name,
            "shots": self.shots,
            "ampx": self.ampx,
            "t_vec": self.t_vec,
            "meta_dict": self.meta_dict,
        }

        self.config = config
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        shots = self.shots
        ampx = self.ampx
        pulse_name = self.pulse_name
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
                with for_each_(t, self.t_vec):
                    play(pulse_name * amp(ampx), qubit, duration=t)

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
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).average().save("ss")

        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss"])
        ss_array = result.fetch_all()[0]
        self.data.update(
            {
                "ss_data": ss_array,
            }
        )

        plt.figure()
        try:
            fit = Fit()
            rabi_fit = fit.rabi(self.t_vec * u.ns, ss_array, plot=True)
            plt.title(f"Time Rabi")
            plt.xlabel("Qubit pulse duration [ns]")
            plt.ylabel("Excited state population")
            x_180_len = round(1 / rabi_fit["f"][0] / 2)
            rabi_freq_MHz = rabi_fit["f"][0] * 1e3
            print(f"Optimal x180_len = {x_180_len} ns for ampx = {self.ampx:.3f}")
            print(f"Rabi frequency = {rabi_freq_MHz:.5f} Hz for ampx = {self.ampx:.3f}")

            plt.title(
                f"Pi pulse len: {x_180_len} ns, Rabi freq: {rabi_freq_MHz:.5f} MHz, ampx: {self.ampx:.3f}"
            )
        except:
            print("Fit Failed")
        plt.plot(self.t_vec * u.ns, ss_array, "o")
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Excited state population")
        plt.grid()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()
