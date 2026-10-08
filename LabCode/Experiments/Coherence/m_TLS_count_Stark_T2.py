from qm.qua import *
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential, T1_seed
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class ACStarkTLST2(ExperimentClass):
    """
    AC Stark T1 experiment
    """
    def __init__(self,element,shots, tau, amp_vec, phase_vec, stark_detuning, calib_params,outer_folder='',path='T1', suffix='data', meta_dict=None, config=None, num_pulses=1,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config=config
        val = self.config["waveforms"]["ps_wf_q2"]["sample"]
        assert isinstance(val, (float, int)), f"Invalid sample type: {type(val)}"
        self.num_pulses = num_pulses
        self.element = element #meta dict element
        self.tau = tau # units clock cycles
        self.phase_vec = phase_vec
        self.shots = shots
        self.stark_detuning = stark_detuning
        self.amp_vec_abs = amp_vec
        self.amp_vec = self.amp_vec_abs / meta_dict["cw_amp"]
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "tau": self.tau,
            "phase_vec": self.phase_vec,
            'amp_vec': self.amp_vec,
            'shots': self.shots,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        qubit_freq = self.meta_dict['IF_freq']
        stark_detuned_freq = self.meta_dict['IF_freq'] + self.stark_detuning
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            state = declare(int)
            a = declare(fixed)
            p = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_((a, p), (self.amp_vec, self.phase_vec)):
                    update_frequency(qubit, qubit_freq)
                    play('X90', qubit)

                    align(qubit, resonator)
                    update_frequency(qubit, stark_detuned_freq)
                    play("constant_tone" * amp(a), qubit)
                    align(qubit, resonator)
                    update_frequency(qubit, qubit_freq)

                    frame_rotation_2pi(p, qubit)
                    play('Y180', qubit)

                    align(qubit, resonator)
                    update_frequency(qubit, stark_detuned_freq)
                    play("constant_tone" * amp(a), qubit)
                    align(qubit, resonator)
                    update_frequency(qubit, qubit_freq)
                    frame_rotation_2pi(p, qubit)
                    play('X90', qubit)
                    reset_frame()
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
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)

            with stream_processing():
                ss_stream.buffer(len(self.amp_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")

        # --- Setup figure ---
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))

        line, = ax.plot([], [], '.-', lw=2)
        ax.set_xlabel("Stark amplitude (arb)")
        ax.set_ylabel("Population ")
        ax.set_title("AC Stark TLS Count")
        ax.grid(True)

        # Fix x-axis (doesn't change)
        x = self.amp_vec_abs  # or self.amp_vec depending on what you want displayed
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, 1)

        while results.is_processing():
            ss_data = results.fetch_all()

            if ss_data is None:
                continue

            # Update plot
            line.set_data(x, ss_data)

            # Optional: auto-scale y if needed
            # ax.set_ylim(0, max(1, np.nanmax(y)*1.1))

            ax.relim()
            ax.autoscale_view(scaley=False)

            plt.pause(0.1)

        plt.ioff()
        #plt.show()

        # Save into data dict
        self.data["ss_data"] = ss_data
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)