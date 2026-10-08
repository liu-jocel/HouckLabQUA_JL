from qm.qua import *
from qm import QuantumMachinesManager
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.RBSupport import RBAnalysis
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
import random as rd
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.loops import from_array
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

from qm import SimulationConfig


class DRAGOrbit(ExperimentClass):
    '''
    RB experiement

    '''

    def __init__(self, element, alpha_vec, circuit_depths, num_circuits, shots, calib_params, outer_folder='', path='',
                 suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.meta_dict = meta_dict
        self.config = config
        self.alpha_vec = alpha_vec
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1000000)

        self.data = {
            'meta_dict': meta_dict,
            'config': config,
            'alpha_vec': alpha_vec,
            'circuit_depths': circuit_depths,
            'num_circuts': num_circuits,
            'shots': shots,
            'seed': self.seed
        }
        self.acquire_RB()
        self.analyze()
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        element = self.element
        resonator = resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['global_params']['reset_time']

        circuit_depths = self.circuit_depths
        max_circuit_depth = np.max(circuit_depths)
        num_circuits = self.num_circuits
        shots = self.shots
        seed = self.seed

        with program() as prog:
            depth = declare(int)  # QUA variable for the varying depth
            # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
            saved_gate = declare(int)
            a = declare(fixed)
            k = declare(int)  # QUA variable for the loop over random sequences
            n = declare(int)  # QUA variable for the averaging loop
            I = declare(fixed)  # QUA variable for the 'I' quadrature
            Q = declare(fixed)  # QUA variable for the 'Q' quadrature
            # The relevant streams
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(k, 0, k < num_circuits, k + 1):  # QUA for_ loop over the random sequences
                sequence_list, inv_gate_list = generate_sequence(seed,
                                                                 max_circuit_depth)  # Generate the random sequence of length max_circuit_depth
                with for_(*from_array(a, self.alpha_vec)):
                    with for_each_(depth, self.circuit_depths):  # Loop over the depths
                        # Replacing the last gate in the sequence with the sequence's inverse gate
                        # The original gate is saved in 'saved_gate' and is being restored at the end
                        assign(saved_gate, sequence_list[depth])
                        assign(sequence_list[depth], inv_gate_list[depth - 1])

                        with for_(n, 0, n < shots, n + 1):  # Averaging loop
                            # Can be replaced by active reset
                            wait(reset_time * u.ns, element)
                            align(element, resonator)
                            # Align the two elements to play the sequence after qubit initialization
                            # The strict_timing ensures that the sequence will be played without gaps
                            # Play the random sequence of desired depth
                            play_sequence_drag_cal(element, a, sequence_list, depth)
                            align(element, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            # Align the two elements to measure after playing the circuit.
                            save(I, I_stream)
                            save(Q, Q_stream)
                            assign(sequence_list[depth], saved_gate)

            with stream_processing():
                I_stream.save_all('I')
                Q_stream.save_all('Q')
                # seq_stream.save_all('circuits')=`
            return prog

    def acquire_RB(self):

        self.job = self.acquire(self.make_prog(), self.config)

        res_handles = self.job.result_handles
        I_ss_handle = res_handles.get('I')
        Q_ss_handle = res_handles.get('Q')
        # circuit_handle = res_handles.get('circuits')

        I_ss_handle.wait_for_all_values()
        Q_ss_handle.wait_for_all_values()
        # circuit_handle.wait_for_all_values()
        I_ss_array = I_ss_handle.fetch_all()['value']
        Q_ss_array = Q_ss_handle.fetch_all()['value']
        # self.circuits = circuit_handle.fetch_all()['value']
        print('QUA program results have been returned')

        raw_data = (I_ss_array, Q_ss_array)
        self.data.update({'raw_data_IQ': raw_data})

    def analyze(self):

        ss_arrays = super().job_to_shots_1q(self.job, [len(self.alpha_vec), self.num_circuits, len(self.circuit_depths),
                                                       self.shots], self.calib_params)
        fit_arrays = ss_arrays.mean(axis=(1, 3))
        return_probs = 1 - fit_arrays
        self.data.update({'single_shot': ss_arrays,
                          'return_prob': return_probs})

        def fitRB(x, a, p, b):
            return a * p ** x + b

        fidelities = []
        errors = []
        for i, rb in enumerate(return_probs):
            try:
                popt, pcov = curve_fit(fitRB, self.circuit_depths, rb,
                                       [0.3, 0.99, 0.5], maxfev=5000)
                a = popt[0]
                p = popt[1]
                p_err = np.sqrt(pcov[1][1])
                b = popt[2]
                self.data['data']['fit'] = popt, pcov
                gate_fidel = 1 - 0.5 * (1 - p) / 1.875  # There are an average of 1.875 physical gates per Clifford
                gate_fidel_err = p_err * 0.5 / 1.875
                self.amp = a
                self.off = b
                fidelities.append(gate_fidel)
                errors.append(gate_fidel_err)
            except:
                fidelities.append(0)
                errors.append(0)

        max_fidel = max(fidelities)
        max_fidel_index = fidelities.index(max_fidel)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(self.alpha_vec, fidelities, 'o')
        plt.xlabel("Drag Alpha")
        plt.ylabel("RB Fidelity")
        plt.title(
            f"{self.element} Drag Calibration. Max Fidelity: {max_fidel: .4f}, alpha: {self.alpha_vec[max_fidel_index]}")
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()

        self.data.update({
            "fidelities": (fidelities, errors),
        })


def generate_sequence(seed, longest_circuit):
    '''
    Helper Function to create RB sequences
    '''
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=longest_circuit + 1)
    inv_gate = declare(int, size=longest_circuit + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < longest_circuit, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence_drag_cal(element, alpha, sequence_list, depth):
    '''
    Helper Function to play a RB sequence
    '''
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(25, element)
            with case_(1):
                play('X180' * amp(1, 0, 0, alpha), element)
            with case_(2):
                play('Y180' * amp(1, 0, 0, alpha), element)
            with case_(3):
                play('Y180' * amp(1, 0, 0, alpha), element)
                play('X180' * amp(1, 0, 0, alpha), element)

            with case_(4):
                play('X90' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)

            with case_(5):
                play('X90' * amp(1, 0, 0, alpha), element)
                play('Ym90' * amp(1, 0, 0, alpha), element)

            with case_(6):
                play('Xm90' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)

            with case_(7):
                play('Xm90' * amp(1, 0, 0, alpha), element)
                play('Ym90' * amp(1, 0, 0, alpha), element)

            with case_(8):
                play('Y90' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(9):
                play('Y90' * amp(1, 0, 0, alpha), element)
                play('Xm90' * amp(1, 0, 0, alpha), element)

            with case_(10):
                play('Ym90' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(11):
                play('Ym90' * amp(1, 0, 0, alpha), element)
                play('Xm90' * amp(1, 0, 0, alpha), element)

            with case_(12):
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(13):
                play('Xm90' * amp(1, 0, 0, alpha), element)

            with case_(14):
                play('Y90' * amp(1, 0, 0, alpha), element)

            with case_(15):
                play('Ym90' * amp(1, 0, 0, alpha), element)

            with case_(16):
                play('Xm90' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(17):
                play('Xm90' * amp(1, 0, 0, alpha), element)
                play('Ym90' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(18):
                play('X180' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)

            with case_(19):
                play('X180' * amp(1, 0, 0, alpha), element)
                play('Ym90' * amp(1, 0, 0, alpha), element)

            with case_(20):
                play('Y180' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(21):
                play('Y180' * amp(1, 0, 0, alpha), element)
                play('Xm90' * amp(1, 0, 0, alpha), element)

            with case_(22):
                play('X90' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)
                play('X90' * amp(1, 0, 0, alpha), element)

            with case_(23):
                play('Xm90' * amp(1, 0, 0, alpha), element)
                play('Y90' * amp(1, 0, 0, alpha), element)
                play('Xm90' * amp(1, 0, 0, alpha), element)