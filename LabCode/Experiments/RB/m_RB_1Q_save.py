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
from qm import SimulationConfig

class RB1Q(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.max_depth = 2 * max(circuit_depths)
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1000000)

        self.data = {
            "meta_dict": meta_dict,
            'config': config,
            "circuit_depths": circuit_depths,
            "num_circuts": num_circuits,
            "shots": shots,
            'seed': self.seed
        }
        self.acquire_RB()
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()
        

    def make_prog(self):
        reset_ticks = self.meta_dict['reset_ticks']
        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            s = declare(int)
            depth_vec = declare(int, value=self.circuit_depths)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            seq_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.max_depth)

                with for_(s, 0, s < self.max_depth, s + 1):
                    save(sequence_list[s], seq_stream)

                with for_each_(m, self.circuit_depths):
                    # m_current = depth_vec[m]
                    # m_pow = m_pow * 2
                    assign(saved_gate, sequence_list[m])
                    assign(sequence_list[m], inv_gate_list[m - 1])
                    with for_(n, 0, n < self.shots, n + 1):
                        wait(reset_ticks, self.element)
                        align()
                        play_sequence(self.element, sequence_list, m)
                        align()
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                    assign(sequence_list[m], saved_gate)
            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
                seq_stream.save_all("circuits")
        return prog

    def acquire_RB(self):

        self.job = self.acquire(self.make_prog(), self.config)

        res_handles = self.job.result_handles
        I_ss_handle = res_handles.get("I")
        Q_ss_handle = res_handles.get("Q")
        circuit_handle = res_handles.get("circuits")

        I_ss_handle.wait_for_all_values()
        Q_ss_handle.wait_for_all_values()
        circuit_handle.wait_for_all_values()
        I_ss_array = I_ss_handle.fetch_all()['value']
        Q_ss_array = Q_ss_handle.fetch_all()['value']
        self.circuits = circuit_handle.fetch_all()['value']
        print("QUA program results have been returned")

        raw_data = (I_ss_array, Q_ss_array, )
        self.data.update({'raw_data_IQ': raw_data, 'sequences': self.circuits})

    def analyze(self):

        ss_array = super().job_to_shots_1q(self.job, [self.num_circuits, len(self.circuit_depths), self.shots], self.calib_params)
        fit_array = ss_array.mean(axis=(0, 2))
        return_prob = 1 - fit_array
        self.data.update({'single_shot': ss_array, 
                          'return_prob': return_prob})

        def fitRB(x, a, p, b):
            return a * p ** x + b

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(self.circuit_depths, return_prob, 'o')
        
        # try:
        popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                           [0.4, 0.99, 0.5], maxfev=5000)

        a = popt[0]
        p = popt[1]
        p_err = np.sqrt(pcov[1][1])
        b = popt[2]
        self.data.update({'fit_popt': popt, 'fit_pcov': pcov})
        clif_fidel = 1 - 0.5 * (1 - p)
        gate_fidel = 1 - 0.5 * (1 - p) / 1.875  # There are an average of 1.875 physical gates per Clifford
        gate_fidel_err = p_err * 0.5 / 1.875
        self.amp = a
        self.off = b

        self.data.update({'RB_fidelity': gate_fidel})
        ax.set_title(f'RB on {self.element}, F = {gate_fidel:.4f} pm {gate_fidel_err:.4f}')
        ax.plot(np.linspace(self.circuit_depths[0], self.circuit_depths[-1]),
                fitRB(np.linspace(self.circuit_depths[0], self.circuit_depths[-1]), *popt))

        print(f'1Q RB fidelity: {gate_fidel}')
        self.data.update({
            'gate_fidelity': gate_fidel,
            'gate_fidelity_error': gate_fidel_err,
            'clifford_fidelity': clif_fidel,
            'fit_popt': popt, 'fit_pcov': pcov
        })
        # except:
        ax.set_title(f'RB on {self.element}')
        print("RB Fit Failed.")
        ax.set_xlabel('Num Cliffords')
        ax.set_ylabel('Return probability')
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

        return ss_array
    

def generate_sequence(seed, longest_circuit):
    """
    Helper Function to create RB sequences
    """
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
        



def play_sequence(element, sequence_list, depth):
    """
    Helper Function to play a RB sequence
    """
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(25, element)
            with case_(1):
                play("X180", element)
            with case_(2):
                play("Y180", element)
            with case_(3):
                play("Y180", element)
                play("X180", element)

            with case_(4):
                play("X90", element)
                play("Y90", element)

            with case_(5):
                play("X90", element)
                play("Ym90", element)

            with case_(6):
                play("Xm90", element)
                play("Y90", element)

            with case_(7):
                play("Xm90", element)
                play("Ym90", element)

            with case_(8):
                play("Y90", element)
                play("X90", element)

            with case_(9):
                play("Y90", element)
                play("Xm90", element)

            with case_(10):
                play("Ym90", element)
                play("X90", element)

            with case_(11):
                play("Ym90", element)
                play("Xm90", element)

            with case_(12):
                play("X90", element)

            with case_(13):
                play("Xm90", element)

            with case_(14):
                play("Y90", element)

            with case_(15):
                play("Ym90", element)

            with case_(16):
                play("Xm90", element)
                play("Y90", element)
                play("X90", element)

            with case_(17):
                play("Xm90", element)
                play("Ym90", element)
                play("X90", element)

            with case_(18):
                play("X180", element)
                play("Y90", element)

            with case_(19):
                play("X180", element)
                play("Ym90", element)

            with case_(20):
                play("Y180", element)
                play("X90", element)

            with case_(21):
                play("Y180", element)
                play("Xm90", element)

            with case_(22):
                play("X90", element)
                play("Y90", element)
                play("X90", element)

            with case_(23):
                play("Xm90", element)
                play("Y90", element)
                play("Xm90", element)