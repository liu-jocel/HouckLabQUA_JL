from qm.qua import *
from qm import QuantumMachinesManager
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.RBSupport import RBAnalysis
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.CR_2QRB_Helpers.TwoQ_RB_Sequence_Generation_CNOT_CR import *
from scipy.optimize import curve_fit
from qualang_tools.results import progress_counter, fetching_tool
from qm import SimulationConfig
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class RB2Q_CNOT(ExperimentClass):
    """
    RB experiement
    """
    def __init__(self, circuit_depths, num_circuits, shots, calib_params, confusion, interleaved=False, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None,simulate=False):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion
        
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.interleaved = interleaved
        if interleaved:
            self.sequence_list, self.len_list = pre_generate_sequence_interleaved(self.num_circuits, self.circuit_depts)
        else:
            self.sequence_list, self.len_list = pre_generate_sequence(self.num_circuits, self.circuit_depts)
        

        self.q1 = self.meta_dict['q1']['q_name']
        self.q2 = self.meta_dict['q2']['q_name']
        self.qc = self.meta_dict['qc']['q_name']
        self.qt = self.meta_dict['qt']['q_name']
        self.r1 = self.meta_dict['q1']['r_name']
        self.r2 = self.meta_dict['q2']['r_name']

        self.data = {
            "meta_dict": meta_dict,
            'config': config,
            "circuit_depths": circuit_depths,
            "num_circuts": num_circuits,
            "shots": shots,
            'sequence_list': self.sequence_list
        }
        if simulate:
            self.simulate(self.self.make_prog(), self.config)
        else:
            self.job = self.acquire(self.make_prog(), self.config)
            self.analyze()
            self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            self.h5_data()
        

    def make_prog(self):
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        resonator_dump_time = self.meta_dict["q1"]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]

        with program() as rb:
            depth = declare(int)  # QUA variable for the varying depth_list index
            depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
            depth_len = declare(int)
            assign(depth_len, len(self.circuit_depts))
            n_avg_ = declare(int, value=shots)
            m = declare(int)  # QUA variable for the loop over random sequences
            n = declare(int)  # QUA variable for the averaging loop
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)

            state1 = declare(bool)  # QUA variable for state discrimination
            # state2 = declare(bool)
            # The relevant streams
            m_st = declare_stream()

            state1_st = declare_stream()
            state2_st = declare_stream()

            sequence_qua = declare(int, value=self.sequence_list)
            len_list_qua = declare(int, value=self.len_list)
            num_of_sequences_ = declare(int, self.num_circuits)

            start = declare(int, value=0)
            run = declare(int, value=0)

            with for_(m, 0, m < self.num_circuits, m + 1):  # QUA for_ loop over the random sequences

                # No flux element in this experiment
                align()
                with for_(depth, 0, depth < len(self.circuit_depths), depth + 1):  # Loop over the depths

                    # Insert the last gate in the sequence by the sequence's inverse gate
                    # Only played the depth corresponding to target_depth

                    with for_(n, 0, n < n_avg_, n + 1):
                        align('q1', 'q2', 'qc', 'qt', 'r1', 'r2')
                        align('q1', 'q2', 'qc', 'qt', 'r1', 'r2')

                        play_sequence(sequence_qua, start, len_list_qua[run])
                        align('q1', 'q2', 'qc', 'qt', 'r1', 'r2')
                        # Align the two elements to measure after playing the circuit.
                        # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                        I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                        assign(I_reset_1, I1)
                        assign(I_reset_2, I2)

                        # State Discrimination
                        assign(state1, (I1 < ss_thresh_q1) & (I2 < ss_thresh_q2))
                        save(state1, state1_st)

                        # Active reset
                        with while_((I_reset_1 > ground_confidence_threshold_q1)
                                    | (I_reset_2 > ground_confidence_threshold_q2)):
                            with if_(I_reset_1 > ss_thresh_q1):
                                play('X180', 'q1',)
                            with if_(I_reset_2 > ss_thresh_q2):
                                play('X180', 'q2', )
                            align('q1', 'q2', 'qc', 'qt', 'r1', 'r2')
                            I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                            assign(I_reset_1, I1)
                            assign(I_reset_2, I2)
                        wait(resonator_dump_time * u.ns, 'q1')

                        assign(start, start + len_list_qua[run])
                        assign(run, run + 1)
                        # Go to the next depth
                        align()
                save(m, m_st)

            with stream_processing():
                m_st.save("iteration")
                # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                state1_st.boolean_to_int().buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depts)).buffer(
                    self.num_circuits
                ).save("state1")
                # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
                state1_st.boolean_to_int().buffer(self.shots).map(FUNCTIONS.average()).buffer(
                    len(self.circuit_depths)).average().save(
                    "state1_avg"
                )
    def analyze(self):
        fetching_list = ["iteration", "state1", "state1_avg"]
        results = fetching_tool(self.job, data_list=["ss", 'circuits'])

        try:
            while results.is_processing():
                # Fetch results
                m, return_prob, shots = results.fetch_all()
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.plot(self.circuit_depths, return_prob, 'o')
                popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                       [0.4, 0.99, 0.5], maxfev=5000)

                a = popt[0]
                p = popt[1]
                p_err = np.sqrt(pcov[1][1])
                b = popt[2]
                self.data.update({'fit_popt': popt, 'fit_pcov': pcov})
                clif_fidel = 1 - 0.5 * (1 - p)

                self.amp = a
                self.off = b
                ax.set_title(f'RB on {self.element}, F = {clif_fidel:.6f} pm {p_err:.6f}')
                ax.plot(np.linspace(self.circuit_depths[0], self.circuit_depths[-1]),
                        fitRB(np.linspace(self.circuit_depths[0], self.circuit_depths[-1]), *popt))

                print(f'2Q RB fidelity: {clif_fidel}')
                self.data.update({
                    'clifford_fidelity_error': p_err,
                    'clifford_fidelity': clif_fidel,
                    'fit_popt': popt, 'fit_pcov': pcov})
            plt.savefig(self.iname[:-4], bbox_inches='tight')

            # Save results
            self.h5_data()
        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            print(f'2Q RB fidelity: {self.data["clifford_fidelity"]} +- {self.data["clifford_fidelity_error"]}')

def fitRB(x, a, p, b):
    return a * p ** x + b

def play_sequence(sequence, start, length):
    align('q1', 'q2', 'qc', 'qt')
    i = declare(int)
    align('q1', 'q2', 'qc', 'qt')
    with for_(i, start, i < start + length, i + 1):
        align('q1', 'q2', 'qc', 'qt')
        with switch_(sequence[i], unsafe=False):
            with case_(0):
                # identity
                align('q1', 'q2', 'qc', 'qt')
            with case_(1):
                play("X90", 'q1')
            with case_(2):
                play("Xm90", 'q1')
            with case_(3):
                play("X180", 'q1')
            with case_(4):
                play("Y90", 'q1')
            with case_(5):
                play("Ym90", 'q1')
            with case_(6):
                play("Y180", 'q1')
            with case_(7):
                play("X90", 'q2')
            with case_(8):
                play("X90", 'q1')
                play("X90", 'q2')
            with case_(9):
                play("Xm90", 'q1')
                play("X90", 'q2')
            with case_(10):
                play("X180", 'q1')
                play("X90", 'q2')
            with case_(11):
                play("Y90", 'q1')
                play("X90", 'q2')
            with case_(12):
                play("Ym90", 'q1')
                play("X90", 'q2')
            with case_(13):
                play("Y180", 'q1')
                play("X90", 'q2')
            with case_(14):
                play("Xm90", 'q2')
            with case_(15):
                play("X90", 'q1')
                play("Xm90", 'q2')
            with case_(16):
                play("Xm90", 'q1')
                play("Xm90", 'q2')
            with case_(17):
                play("X180", 'q1')
                play("Xm90", 'q2')
            with case_(18):
                play("Y90", 'q1')
                play("Xm90", 'q2')
            with case_(19):
                play("Ym90", 'q1')
                play("Xm90", 'q2')
            with case_(20):
                play("Y180", 'q1')
                play("Xm90", 'q2')
            with case_(21):
                play("X180", 'q2')
            with case_(22):
                play("X90", 'q1')
                play("X180", 'q2')
            with case_(23):
                play("Xm90", 'q1')
                play("X180", 'q2')
            with case_(24):
                play("X180", 'q1')
                play("X180", 'q2')
            with case_(25):
                play("Y90", 'q1')
                play("X180", 'q2')
            with case_(26):
                play("Ym90", 'q1')
                play("X180", 'q2')
            with case_(27):
                play("Y180", 'q1')
                play("X180", 'q2')
            with case_(28):
                play("Y90", 'q2')
            with case_(29):
                play("X90", 'q1')
                play("Y90", 'q2')
            with case_(30):
                play("Xm90", 'q1')
                play("Y90", 'q2')
            with case_(31):
                play("X180", 'q1')
                play("Y90", 'q2')
            with case_(32):
                play("Y90", 'q1')
                play("Y90", 'q2')
            with case_(33):
                play("Ym90", 'q1')
                play("Y90", 'q2')
            with case_(34):
                play("Y180", 'q1')
                play("Y90", 'q2')
            with case_(35):
                play("Ym90", 'q2')
            with case_(36):
                play("X90", 'q1')
                play("Ym90", 'q2')
            with case_(37):
                play("Xm90", 'q1')
                play("Ym90", 'q2')
            with case_(38):
                play("X180", 'q1')
                play("Ym90", 'q2')
            with case_(39):
                play("Y90", 'q1')
                play("Ym90", 'q2')
            with case_(40):
                play("Ym90", 'q1')
                play("Ym90", 'q2')
            with case_(41):
                play("Y180", 'q1')
                play("Ym90", 'q2')
            with case_(42):
                play("Y180", 'q2')
            with case_(43):
                play("X90", 'q1')
                play("Y180", 'q2')
            with case_(44):
                play("Xm90", 'q1')
                play("Y180", 'q2')
            with case_(45):
                play("X180", 'q1')
                play("Y180", 'q2')
            with case_(46):
                play("Y90", 'q1')
                play("Y180", 'q2')
            with case_(47):
                play("Ym90", 'q1')
                play("Y180", 'q2')
            with case_(48):
                play("Y180", 'q1')
                play("Y180", 'q2')
            with case_(49):
                # Echo CR code here (in progress...)
                frame_rotation_2pi(+0.25, 'q1')  # +0.25 for Z(-pi/2)
                play("Xm90", 'q2')

                # Shift frames to the calibrated phases
                # Play CR
                align("q1", "q2", "qc", "qt", "r1", "r2")
                # CR Echo
                play("ZX45", "qc",)
                play("ZX45", "qt",)
                align("q1", "q2", "qc", "qt")
                play("X180", "q1")
                align("q1", "q2", "qc", "qt")
                play("ZXm45", "qc",)
                play("ZXm45", "qt",)
                align("q1", "q2", "qc", "qt")
                play("X180", "q1")
                align("q1", "q2", "qc", "qt")

                # Shift frames to the calibrated phases
                frame_rotation_2pi(cr_drive_phase_ZI_correct, 'qc')
                align("q1", "q2", "qc", "qt", "r1", "r2")
                frame_rotation_2pi(+0.25, 'q1')
                # check the sign before 0.25 -- could be either way depending on 'qc'_phase
                play("Y90" * amp(0.00001), 'q1', duration=1)
                # this is a tricky step. the goal is to avoid face-to-face frame rotations
                # qm only recognizes the last frame rotation before a real pulse if multiple frame rotations are stacked
                # e.g., two CNOTs can be adjacent in a random gate sequence, and two 0.25 rotations are intended
                # but qm only implements the last 0.25 rotation

                # if you want to avoid this subtlety, you can use real XY gates to replace the virtual Z/2
                # Z/2: play('Xm90', 'q1'); play('Y90', 'q1'); play('X90', 'q1');
                # -Z/2: play('Xm90', 'q1'); play('Ym90', 'q1'); play('X90', 'q1');

                align("q1", "q2", "qc", "qt", "r1", "r2")
        

