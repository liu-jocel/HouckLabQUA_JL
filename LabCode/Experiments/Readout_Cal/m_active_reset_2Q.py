from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.SPAM_correction import correct_counts_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class ActiveResetCal2Q(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """
    def __init__(self, t_vec, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
                 suffix='data',):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion
        self.read_len = self.meta_dict['global_params']['read_len']

        self.q1 = self.meta_dict['q1']['q_name']
        self.q2 = self.meta_dict['q2']['q_name']
        self.qc = self.meta_dict['qc']['q_name']
        self.qt = self.meta_dict['qt']['q_name']
        self.r1 = self.meta_dict['q1']['r_name']
        self.r2 = self.meta_dict['q2']['r_name']

        self.t_vec = t_vec
        self.data = {
            "shots": self.shots,
            "t_vec": self.t_vec,
            "meta_dict": self.meta_dict,
            'calib_params': self.calib_params,
            'confusion': self.confusion,
        }
        self.job = self.acquire(self.make_prog(), self.config)

        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        """
        Qua Program for Standard CR experimenet
        """
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        q1 = self.q1
        q2 = self.q2
        qc = self.qc
        qt = self.qt
        r1 = self.r1
        r2 = self.r2
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            state_1 = declare(int)
            state_2 = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)
            state_stream_1 = declare_stream()
            state_stream_2 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    play("X180", q1)
                    play("X180", q2)
                    align(q1, q2, qc, qt, r1, r2)
                    I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                    assign(I_reset_1, I1)
                    assign(I_reset_2, I2)

                    # State Discrimination
                    with if_(I1 > ss_thresh_q1):
                        assign(state_1, 1)
                    with else_():
                        assign(state_1, 0)

                    with if_(I2 > ss_thresh_q2):
                        assign(state_2, 1)
                    with else_():
                        assign(state_2, 0)
                    save(state_1, state_stream_1)
                    save(state_2, state_stream_2)
                    align(q1, q2, qc, qt, r1, r2)
                    #Active reset
                    with while_((I_reset_1 > ground_confidence_threshold_q1)
                                | (I_reset_2 > ground_confidence_threshold_q2)):
                        with if_(I_reset_1 > ss_thresh_q1):
                            play('X180', q1, )
                        with if_(I_reset_2 > ss_thresh_q2):
                            play('X180', q2,)
                        align(q1, q2, qc, qt, r1, r2)
                        I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                        assign(I_reset_1, I1)
                        assign(I_reset_2, I2)
                    align(q1, q2, qc, qt, r1, r2)
                    wait(t, q1)

            with stream_processing():
                state_stream_1.buffer(len(self.t_vec)).average().save("ss1")
                state_stream_2.buffer(len(self.t_vec)).average().save("ss2")
        return prog
    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss1", "ss2"])
        excited_probs_q1 = result.fetch_all()[0]
        excited_probs_q2 = result.fetch_all()[1]
        t_vec = self.t_vec / u.us

        # Create a 2x3 grid of subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Flatten the axes array for easy iteration
        axs = axs.flatten()
        meas_results = {
            "Excited Population q1": np.array(excited_probs_q1),
            "Excited Population q2": np.array(excited_probs_q2),
        }
        # Adjust t_vec to compensate for echo in plotting
        for i, (title, data) in enumerate(meas_results.items()):
            # Iterate over the dictionary and plot each key-value pair
            axs[i].plot(t_vec, data, "o", color="black")
            axs[i].set_ylim(0, 1)
            axs[i].set_title(title)
            axs[i].set_xlabel("Thermalization Time (us)")
            axs[i].set_ylabel("Excited State Probability")
        plt.suptitle("Active Reset Calibtation 2Q")
        plt.tight_layout()
        plt.savefig(self.iname[:-4], bbox_inches='tight')
        plt.close(fig)
        self.data.update({
            "qubit_dynamics": meas_results})