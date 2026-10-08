from qm.qua import *
from LabCode.CoreLib.experiment import ExperimentClass
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


def measure_1q(element, I, Q, timestamp_stream=None):
    measure('read', element['r_name'], None, demod.full('integ_W1_' + element['r_name'], I, 'out'),
            demod.full('integ_W2_' + element['r_name'], Q, 'out'), timestamp_stream=timestamp_stream)
    return I, Q


class T1_RAW(ExperimentClass):
    """
    T1 experiment
    """
    def __init__(self, element, t_vec, calib_params, meta_dict=None, config=None):
        super().__init__(meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element # meta dict element
        self.t_vec = get_equivalent_log_array(t_vec)
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec,
            'meta_dict': self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), self.config)

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            ss_stream = declare_stream()
            tau_stream = declare_stream()
            timestamps = declare_stream()

            with infinite_loop_():
                with for_(*from_array(t, self.t_vec * u.ns)):
                    play('X180', qubit)
                    wait(t, qubit)

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

                    # # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)

            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).save_all("ss")
                I_stream.buffer(len(self.t_vec)).save_all("I")
                Q_stream.buffer(len(self.t_vec)).save_all("Q")
                timestamps.buffer(len(self.t_vec)).save_all("timestamps")
                tau_stream.buffer(len(self.t_vec)).save("tau")

        return prog
