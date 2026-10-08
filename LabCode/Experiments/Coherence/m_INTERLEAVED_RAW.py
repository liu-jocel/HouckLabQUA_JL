from qm.qua import *
from LabCode.CoreLib.experiment import ExperimentClass
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
import numpy as np
u = unit(coerce_to_integer=True)


def measure_1q(element, I, Q, timestamp_stream=None):
    measure('read', element['r_name'], None, demod.full('integ_W1_' + element['r_name'], I, 'out'),
            demod.full('integ_W2_' + element['r_name'], Q, 'out'), timestamp_stream=timestamp_stream)
    return I, Q


class INTERLEAVED_RAW(ExperimentClass):
    """
    Interleaved T1, T2R, T2E, or T2CPMG experiment
    The interleaving is done at the shot level

    EXPERIMENT MODES:
    -1: T1 measurement
    -2: T2R measurement
    -3: T2E measurement
    X > 0: T2CPMG with X pi-pulses
    """
    def _check_equal_lengths(self, t_vec_dict):
        # Get the length of the first array
        iterator = iter(t_vec_dict.values())
        first_length = len(next(iterator))

        # Compare all remaining lengths
        for key, arr in t_vec_dict.items():
            if len(arr) != first_length:
                raise ValueError(
                    f"Array under key '{key}' has length {len(arr)}, "
                    f"expected {first_length}"
                )

    def _process_t_vec_dict(self, t_vec_dict):
        self._check_equal_lengths(t_vec_dict)
        exp_mode_list = []
        t_vec_dict_proc = {}
        for exp, t_vec in t_vec_dict.items():
            if exp == "T1":
                exp_mode_list.append(-1)
                t_vec_dict_proc[exp] = get_equivalent_log_array(t_vec)
            elif exp == "T2R":
                exp_mode_list.append(-2)
                t_vec_dict_proc[exp] = t_vec
            elif exp == "T2E":
                exp_mode_list.append(-3)
                t_vec_dict_proc[exp] = [int(t/2) for t in get_equivalent_log_array(t_vec)]
            elif exp.startswith("T2CPMG"):
                num_pi = int(exp.split("_", 1)[1])
                exp_mode_list.append(num_pi)
                t_vec_dict_proc[exp] = np.array([int(t/(2*num_pi)) for t in t_vec], dtype=int)
        self._check_equal_lengths(t_vec_dict_proc)

        # convert t_vec to clock cycles
        t_cycles_dict = {}
        for exp, t_vec in t_vec_dict_proc.items():
            t_cycles_dict[exp] = [int(t/4) for t in t_vec_dict_proc[exp]] # each clock cycle is 4 ns

        return exp_mode_list, t_cycles_dict

    def _interleave_lists(self, t_vec_dict):
        t_vec_list = list(t_vec_dict.values())
        result = []
        for items in zip(*t_vec_list):
            result.extend([int(x) for x in items])
        return result

    def __init__(self, element, t_vec_dict, calib_params, detuning=0, meta_dict=None, config=None):
        """
        Parameters:
        element -- used to select which qubit is being measured (e.g. q1, q2, etc.)
        t_vec_dict -- dictionary with pairs (exp, t_vec), exp is T1, T2R, T2E, T2CPMG_X where X is number of pi-pulses,
        t_vec is the list of times for the corresponding experiment
        calib_params -- SS cal parameters
        detuning -- detuning for Ramsey experiment, if Ramsey experiment not performed, this is ignored
        """

        super().__init__(meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element # meta dict element
        self.exp_list = t_vec_dict.keys()

        self.exp_mode_list, self.t_cycles_dict = self._process_t_vec_dict(t_vec_dict) # processed version of t_vec_dict
        self.num_exp = len(self.exp_mode_list)
        self.t_vec_clk = self._interleave_lists(self.t_cycles_dict) # units in clock cycles, TODO: check for off-by-one errors
        self.t_points = len(self.t_vec_clk)
        print(self.t_vec_clk)
        print(self.exp_mode_list)

        self.calib_params = calib_params
        self.detuning = detuning
        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec_clk,
            'meta_dict': self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), self.config)

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        qubit_freq = self.meta_dict['IF_freq']
        detuned_freq = self.meta_dict['IF_freq'] + self.detuning
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        N_t = int(self.t_points)

        with program() as prog:
            t = declare(int)
            two_t = declare(int)
            t_index = declare(int)
            n_pi = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            ss_stream = declare_stream()
            tau_stream = declare_stream()
            timestamps = declare_stream()

            counter = declare(int) # counter to switch between experiment modes
            mode_seq = declare(int, size=self.num_exp)
            t_vec_list = declare(int, size=N_t)

            for k, v in enumerate(self.exp_mode_list):
                assign(mode_seq[k], v)

            for k, v in enumerate(self.t_vec_clk):
                assign(t_vec_list[k], int(v))

            mode = declare(int)

            with infinite_loop_():
                assign(counter, 0)
                
                # for t in self.t_vec:
                #with for_each_(t, t_vec_array):
                with for_(t_index, 0, t_index < N_t, t_index + 1):
                    assign(t, t_vec_list[t_index])
                    assign(mode, mode_seq[counter]) # choose which pulse sequence to run

                    with if_(mode == -1): # T1 experiment
                        play('X180', qubit)
                        wait(t, qubit) # note that the units are wrong here -- need to convert to clock cycles
                        #wait(3 * u.ms, qubit)

                    with if_(mode == -2): # T2R experiment
                        update_frequency(qubit, detuned_freq)
                        play('X90', qubit)
                        wait(t, qubit)
                        play('X90', qubit)
                        update_frequency(qubit, qubit_freq)

                    with if_(mode == -3): # T2E experiment
                        play('X90', qubit)
                        wait(t, qubit) # these are already divided by 2 in time from pre-processing above
                        play('Y180', qubit)
                        wait(t, qubit)
                        play('X90', qubit)

                    with if_(mode > 0): # T2CPMG experiment
                        assign(two_t, 2 * t)
                        play('X90', qubit)
                        wait(t, qubit)
                        # Qua does not like division. adds 400 ns latency
                        with for_(n_pi, 0, n_pi < mode - 1, n_pi + 1): # mode encodes the number of pi pulses
                            play('Y180', qubit)
                            wait(two_t, qubit)
                        play('Y180', qubit)
                        wait(t, qubit)
                        play('mX90', qubit)


                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q, timestamps)

                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream)
                    save(I, I_stream)
                    save(Q, Q_stream)
                    save(t, tau_stream)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                         with if_(I_reset > ss_thresh):
                             play('X180', qubit)
                             align(qubit, resonator)
                         I, _ = measure_1q(self.meta_dict, I, Q)
                         assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
                    #wait(1 * u.us, qubit)

                    # Increment counter to next experiment mode
                    assign(counter, counter + 1)
                    with if_(counter >= self.num_exp):
                        assign(counter, 0)


            with stream_processing():
                ss_stream.buffer(len(self.t_vec_clk)).save_all("ss")
                I_stream.buffer(len(self.t_vec_clk)).save_all("I")
                Q_stream.buffer(len(self.t_vec_clk)).save_all("Q")
                timestamps.buffer(len(self.t_vec_clk)).save_all("timestamps")
                tau_stream.buffer(len(self.t_vec_clk)).save("tau")

        return prog
