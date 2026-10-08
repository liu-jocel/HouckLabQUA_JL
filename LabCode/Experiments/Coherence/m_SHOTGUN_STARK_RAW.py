from qm.qua import *
from LabCode.CoreLib.experiment import ExperimentClass
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
import numpy as np

from math import gcd
from functools import reduce

u = unit(coerce_to_integer=True)


def measure_1q(element, I, Q, timestamp_stream=None):
    measure(
        "read",
        element["r_name"],
        None,
        demod.full("integ_W1_" + element["r_name"], I, "out"),
        demod.full("integ_W2_" + element["r_name"], Q, "out"),
        timestamp_stream=timestamp_stream,
    )
    return I, Q


class SHOTGUN_STARK_RAW(ExperimentClass):
    """
    Interleaved Stark-shifted T1, T2E with regular T2R
    The interleaving is done at the shot level

    EXPERIMENT MODES (note each clock cycle is 4 ns):
    -1: T1 measurement
    -2: T2R measurement
    -3: T2E measurement
    -4: T1 Stark
    X > 0: T2CPMG measurement with X pi-pulses
    """

    def _concat_params(self, sweep_vec_dict, amp_max):
        """
        Process the sweep vector to be in clock cycle units (T1/T2R/T2E/T2CPMG) and
        normalized amplitude compared to the max amplitude (T1Stark/T2EStark)

        Creates number-coded experiment and parameter list for OPX control
        """
        mode_list_all = []
        amp_list_all = []
        tau_list_all = []

        for exp, sweep_vec in sweep_vec_dict.items():
            if exp.startswith("T1Stark"):
                tau_ns = int(exp.split("_", 1)[1])

                amp_vec = sweep_vec
                tau_vec = [tau_ns] * len(amp_vec)
                mode_vec = [-4] * len(amp_vec)
            elif exp == "T1":
                tau_vec = get_equivalent_log_array(sweep_vec)
                amp_vec = [0] * len(tau_vec)
                mode_vec = [-1] * len(amp_vec)
            elif exp == "T2R":
                tau_vec = sweep_vec
                amp_vec = [0] * len(tau_vec)
                mode_vec = [-2] * len(tau_vec)
            elif exp == "T2E":
                tau_vec = [int(t / 2) for t in get_equivalent_log_array(sweep_vec)]
                amp_vec = [0] * len(tau_vec)
                mode_vec = [-3] * len(tau_vec)
            elif exp.startswith("T2CPMG"):
                num_pi = int(exp.split("_", 1)[1])
                tau_vec = np.array(
                    [int(t / (2 * num_pi)) for t in sweep_vec], dtype=int
                )
                amp_vec = [0] * len(tau_vec)
                mode_vec = [num_pi] * len(tau_vec)

            amp_list_all.extend(amp_vec)
            tau_list_all.extend(tau_vec)
            mode_list_all.extend(mode_vec)

        # convert tau_list to clock cycles
        tau_list_all_clk = [int(t / 4) for t in tau_list_all]

        # normalize amp_vec to amp_max
        amp_list_all_normalized = [sweep_amp / amp_max for sweep_amp in amp_list_all]

        return mode_list_all, tau_list_all_clk, amp_list_all_normalized

    def __init__(
        self,
        element,
        sweep_vec_dict,
        calib_params,
        stark_t1_detuning=50,
        ramsey_detuning=0,
        amp_max=0,
        meta_dict=None,
        config=None,
    ):
        """
        Parameters:
        element -- used to select which qubit is being measured (e.g. q1, q2, etc.)
        sweep_vec_dict -- dictionary with pairs (exp, sweep_vec), exp is T1, T2R, T2E, T2CPMG_X where X is number of pi-pulses,
        sweep_vec is the list of times for the corresponding experiment
        calib_params -- SS cal parameters
        stark_t1_detuning -- detuning for Stark shift pulse for T1 experiment
        ramsey_detuning -- detuning for Ramsey experiment
        amp_max -- maximum Stark amplitude (for rescaling amplitudes)
        """

        super().__init__(meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element  # meta dict element
        self.exp_list = sweep_vec_dict.keys()

        self.mode_list_all, self.tau_list_all_clk, self.amp_list_all_normalized = (
            self._concat_params(sweep_vec_dict, amp_max)
        )  # processed version of sweep_vec_dict
        self.num_exp = len(self.exp_list)
        assert len(self.tau_list_all_clk) == len(self.amp_list_all_normalized)
        assert len(self.mode_list_all) == len(self.tau_list_all_clk)
        self.num_points = len(self.mode_list_all)

        print(self.meta_dict["cw_amp"])
        print(self.mode_list_all)
        print(self.tau_list_all_clk)
        print(self.amp_list_all_normalized)

        self.calib_params = calib_params
        self.stark_t1_detuning = stark_t1_detuning
        self.ramsey_detuning = ramsey_detuning
        self.data = {
            "qubit": self.element,
            "sweep_vec_amp": self.amp_list_all_normalized,
            "sweep_vec_tau": self.tau_list_all_clk,
            "meta_dict": self.meta_dict,
        }

        self.job = self.acquire(self.make_prog(), self.config)

    def make_prog(self):
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        qubit_freq = self.meta_dict["IF_freq"]
        stark_t1_detuned_freq_pos = self.meta_dict["IF_freq"] + self.stark_t1_detuning
        stark_t1_detuned_freq_neg = self.meta_dict["IF_freq"] - self.stark_t1_detuning
        ramsey_detuned_freq = self.meta_dict["IF_freq"] + self.ramsey_detuning
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        N_x = int(self.num_points)

        with program() as prog:
            a = declare(fixed)
            a_abs = declare(fixed)
            t = declare(int)
            two_t = declare(int)
            x_index = declare(int)
            n_pi = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            # I_stream = declare_stream()
            # Q_stream = declare_stream()
            ss_stream = declare_stream()
            amp_pts_stream = declare_stream()
            tau_pts_stream = declare_stream()
            timestamps = declare_stream()
            exp_order_stream = declare_stream()

            mode_seq = declare(int, size=N_x)
            amp_seq = declare(fixed, size=N_x)
            tau_seq = declare(int, size=N_x)

            mode = declare(int)

            rand = Random()
            i = declare(int)
            j = declare(int)
            temp_int = declare(int)
            temp_fixed = declare(fixed)

            for k, v in enumerate(self.mode_list_all):
                assign(mode_seq[k], v)

            for k, v in enumerate(self.amp_list_all_normalized):
                assign(amp_seq[k], v)

            for k, v in enumerate(self.tau_list_all_clk):
                assign(tau_seq[k], v)

            with infinite_loop_():
                with for_(i, N_x - 1, i > 0, i - 1):
                    assign(j, rand.rand_int(i + 1))  # j in [0, i]

                    # swap values
                    assign(temp_int, mode_seq[i])
                    assign(mode_seq[i], mode_seq[j])
                    assign(mode_seq[j], temp_int)

                    assign(temp_fixed, amp_seq[i])
                    assign(amp_seq[i], amp_seq[j])
                    assign(amp_seq[j], temp_fixed)

                    assign(temp_int, tau_seq[i])
                    assign(tau_seq[i], tau_seq[j])
                    assign(tau_seq[j], temp_int)

                # for t in self.sweep_vec:
                # with for_each_(t, sweep_vec_array):
                with for_(x_index, 0, x_index < N_x, x_index + 1):
                    assign(a, amp_seq[x_index])
                    assign(t, tau_seq[x_index])
                    assign(
                        mode, mode_seq[x_index]
                    )  # choose which pulse sequence to run

                    with if_(mode == -4):  # T1 Stark experiment
                        with if_(a > 0):
                            update_frequency(qubit, qubit_freq)
                            play("X180", qubit)

                            align(qubit, resonator)
                            update_frequency(qubit, stark_t1_detuned_freq_pos)
                            play(
                                "constant_tone" * amp(a), qubit, duration=t
                            )  # param is in units of clock cycles
                            align(qubit, resonator)
                            update_frequency(qubit, qubit_freq)

                        with if_(a < 0):
                            assign(a_abs, -a)
                            update_frequency(qubit, qubit_freq)
                            play("X180", qubit)

                            align(qubit, resonator)
                            update_frequency(qubit, stark_t1_detuned_freq_neg)
                            play(
                                "constant_tone" * amp(a_abs), qubit, duration=t
                            )  # param is in units of clock cycles
                            align(qubit, resonator)
                            update_frequency(qubit, qubit_freq)

                    with if_(mode == -1):  # T1 experiment
                        play("X180", qubit)
                        wait(
                            t, qubit
                        )  # note that the units are wrong here -- need to convert to clock cycles
                        # wait(3 * u.ms, qubit)

                    with if_(mode == -2):  # T2R experiment
                        update_frequency(qubit, ramsey_detuned_freq)
                        play("X90", qubit)
                        wait(t, qubit)
                        play("X90", qubit)
                        update_frequency(qubit, qubit_freq)

                    with if_(mode == -3):  # T2E experiment
                        play("X90", qubit)
                        wait(
                            t, qubit
                        )  # these are already divided by 2 in time from pre-processing above
                        play("Y180", qubit)
                        wait(t, qubit)
                        play("X90", qubit)

                    with if_(mode > 0):  # T2CPMG experiment
                        assign(two_t, 2 * t)
                        play("X90", qubit)
                        wait(t, qubit)
                        # Qua does not like division. adds 400 ns latency
                        with for_(
                            n_pi, 0, n_pi < mode - 1, n_pi + 1
                        ):  # mode encodes the number of pi pulses
                            play("Y180", qubit)
                            wait(two_t, qubit)
                        play("Y180", qubit)
                        wait(t, qubit)
                        play("X90", qubit)

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q, timestamps)

                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream)
                    # save(I, I_stream)
                    # save(Q, Q_stream)
                    save(a, amp_pts_stream)
                    save(t, tau_pts_stream)
                    save(mode, exp_order_stream)
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
                ss_stream.buffer(self.num_points).save_all("ss")
                # I_stream.buffer(self.num_points).save_all("I")
                # Q_stream.buffer(self.num_points).save_all("Q")
                timestamps.buffer(self.num_points).save_all("timestamps")
                amp_pts_stream.buffer(self.num_points).save_all("amp_pts")
                tau_pts_stream.buffer(self.num_points).save_all("tau_pts")
                exp_order_stream.buffer(self.num_points).save_all("exp_order")

        return prog
