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


class TimeRabiCR(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """
    def __init__(self, t_vec, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
                 suffix='data', cancel=False, echo=False):
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
            'cancel': cancel,
            'echo': echo
        }
        self.cancel = cancel
        self.echo = echo

        if self.cancel and self.echo:
            self.job = self.acquire(self.make_prog_echo_cancel(), self.config)
            # compensate t_vec for two ZX45 pulses
            self.t_vec = t_vec/2
        elif self.cancel:
            self.job = self.acquire(self.make_prog_cancel(), self.config)
        elif self.echo:
            self.job = self.acquire(self.make_prog_echo(), self.config)
            # compensate t_vec for two ZX45 pulses
            self.t_vec = t_vec / 2
        else:
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
        resonator_dump_time = self.meta_dict[q1]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            control_state = declare(int) # 0 or 1, State of control qubit
            proj = declare(int) # 0, 1, 2, For projection onto X, Y, Z axis
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
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # CR
                            play("ZX90", qc, duration=t)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
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
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_1.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
                state_stream_2.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss2")
        return prog

    def make_prog_cancel(self):
        """
        Qua program for CR experiment with compensation pulse on qt
        """
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        q1 = self.q1
        q2 = self.q2
        qc = self.qc
        qt = self.qt
        r1 = self.r1
        r2 = self.r2
        resonator_dump_time = self.meta_dict[q1]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]
        with program() as prog:
            n = declare(int)
            t = declare(int)
            control_state = declare(int) # 0 or 1, State of control qubit
            proj = declare(int) # 0, 1, 2, For projection onto X, Y, Z axis
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
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            #Cancel CR
                            play("ZX90", qc, duration=t)
                            play("ZX90", qt, duration=t)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
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

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_1.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
                state_stream_2.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss2")
        return prog

    def make_prog_echo(self):
        """
        Qua program for Echo CR experiment
        """
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        q1 = self.q1
        q2 = self.q2
        qc = self.qc
        qt = self.qt
        r1 = self.r1
        r2 = self.r2
        resonator_dump_time = self.meta_dict[q1]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]
        with program() as prog:
            n = declare(int)
            t = declare(int)
            control_state = declare(int) # 0 or 1, State of control qubit
            proj = declare(int) # 0, 1, 2, For projection onto X, Y, Z axis
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
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Echo CR
                            play("ZX45", qc, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)
                            play("ZXm45", qc, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)

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

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_1.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
                state_stream_2.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss2")
        return prog

    def make_prog_echo_cancel(self):
        """
        Qua program for CR experiment with echo and cancellation tone on qt
        """
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        q1 = self.q1
        q2 = self.q2
        qc = self.qc
        qt = self.qt
        r1 = self.r1
        r2 = self.r2
        resonator_dump_time = self.meta_dict[q1]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            control_state = declare(int)  # 0 or 1, State of control qubit
            proj = declare(int)  # 0, 1, 2, For projection onto X, Y, Z axis
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
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preperation qc
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # CR Echo
                            play("ZX45", qc, duration=t)
                            play("ZX45", qt, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)
                            play("ZXm45", qc, duration=t)
                            play("ZXm45", qt, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
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

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_1.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
                state_stream_2.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss2")
        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss1", "ss2"])
        excited_probs_q1 = result.fetch_all()[0]
        excited_probs_q2 = result.fetch_all()[1]

        meas_results = {
            'X_qc': {
                "qc = 0": excited_probs_q1[:, 0, 0,],
                "qc = 1": excited_probs_q1[:, 0, 1,],
            },
            'X_qt': {
                "qc = 0": excited_probs_q2[:, 0, 0,],
                "qc = 1": excited_probs_q2[:, 0, 1,],
            },
            'Y_qc': {
                "qc = 0": excited_probs_q1[:, 1, 0,],
                "qc = 1": excited_probs_q1[:, 1, 1,],
            },
            'Y_qt': {
                "qc = 0": excited_probs_q2[:, 1, 0,],
                "qc = 1": excited_probs_q2[:, 1, 1,],
            },
            'Z_qc': {
                "qc = 0": excited_probs_q1[:, 2, 0,],
                "qc = 1": excited_probs_q1[:, 2, 1,],
            },
            'Z_qt': {
                "qc = 0": excited_probs_q2[:, 2, 0,],
                "qc = 1": excited_probs_q2[:, 2, 1,],
            },
        }

        if self.echo:
            t_vec = 2 * self.t_vec / u.us
        else:
            t_vec = self.t_vec / u.us

        # Create a 2x3 grid of subplots
        fig, axs = plt.subplots(3, 2, figsize=(10, 12))

        # Flatten the axes array for easy iteration
        axs = axs.flatten()

        # Adjust t_vec to compensate for echo in plotting
        for i, (title, data) in enumerate(meas_results.items()):
            # Iterate over the dictionary and plot each key-value pair
            axs[i].plot(t_vec, data["qc = 0"], "o", color="blue", label="qc = 0")
            axs[i].plot(t_vec, data["qc = 1"], "o", color="red", label="qc = 1")
            axs[i].legend()
            axs[i].set_ylim(0, 1)
            axs[i].set_title(title)
            axs[i].set_xlabel("Time (us)")
            axs[i].set_ylabel("Excited State Probability")

        if self.cancel and self.echo:
            plt.suptitle("Timed CR Echo Cancel Dynamics")
        elif self.cancel:
            plt.suptitle("Timed CR Cancel Dynamics")
        elif self.echo:
            plt.suptitle("Timed CR Echo Dynamics")
        else:
            plt.suptitle("Timed CR Dynamics")

        plt.tight_layout()
        plt.savefig(self.iname[:-4], bbox_inches='tight')
        plt.close(fig)
        self.data.update({
            "qubit_dynamics": meas_results})

    def extract_qt_dynamics(self, data_dict):
        """
        This function traces out qc and returns the probability that qt = 1
        """
        qt_dict = dict()
        for name, data in data_dict.items():
            # returns percentage of shots in "qc, qt" =  "01" + "11"
            qt_dict[name] = (data[1]+data[3]) / self.shots
        return qt_dict

    def extract_qc_dynamics(self, data_dict):
        """
        This function traces out qt and returns the probability that qc = 1
        """
        qc_dict = dict()
        for name, data in data_dict.items():
            # returns percentage of shots in "qc, qt" =  "10" + "11"
            qc_dict[name] = (data[1] + data[3]) / self.shots
        return qc_dict