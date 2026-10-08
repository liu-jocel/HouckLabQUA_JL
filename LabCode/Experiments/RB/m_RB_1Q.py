from qm.qua import *
from qm import QuantumMachinesManager
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.pulse_macros import *
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
import random as rd
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qm import SimulationConfig
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class RB1Q(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None, simulate=False, sim_duration=10_000, calib_params_3state=None,
                 save_h5=True, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        #self.max_depth = 2 * max(circuit_depths)
        self.max_depth = max(circuit_depths)
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        # When provided (list of 3 [I, Q] centroids from SingleShot1Q3Level), the run uses
        # 3-state readout: raw I/Q are saved per shot and classified into |0>/|1>/|2>, and the
        # return probability is renormalized over {0, 1} (|2> shots discarded) so leakage is not
        # misassigned into |1>. Active reset still uses the 2-state thresholds in calib_params.
        self.calib_params_3state = calib_params_3state
        self.use_3state = calib_params_3state is not None
        self.seed = rd.randint(1, 1000000)

        self.data = {
            "meta_dict": meta_dict,
            'config': config,
            "circuit_depths": circuit_depths,
            "num_circuts": num_circuits,
            "shots": shots,
            'seed': self.seed
        }
        if simulate:
            _, _, sim_job = self.simulate(self.make_sim_prog(), self.config, duration=sim_duration)
            self.print_sim_sequence(sim_job)
        else:
            self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
            self.analyze(self.job, self.qm)
            self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            if save_h5:
                self.h5_data()
        

    def make_prog(self):
        if self.use_3state:
            return self.make_prog_3state()
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            # s = declare(int)
            state = declare(int)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            # seq_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.max_depth)
                # with for_(s, 0, s < self.max_depth, s + 1):
                #     save(sequence_list[s], seq_stream)

                with for_each_(m, self.circuit_depths):
                    assign(saved_gate, sequence_list[m])
                    assign(sequence_list[m], inv_gate_list[m - 1])

                    with for_(n, 0, n < self.shots, n + 1):
                        wait(resonator_dump_time*u.ns, self.element)
                        reset_frame(qubit)
                        align()
                        # with strict_timing_():
                        play_sequence(qubit, sequence_list, m)

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
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                    assign(sequence_list[m], saved_gate)
                save(k, k_stream)
            with stream_processing():
                k_stream.save("iteration")
                ss_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depths)).save_all("ss")
                # seq_stream.save_all("circuits")
        return prog

    def make_prog_3state(self):
        """RB program with 3-state readout: identical sequence + active reset as the 2-state
        program, but raw I/Q are streamed per shot (no FPGA thresholding/averaging) so the shots
        can be classified into |0>/|1>/|2> offline. Mirrors the leakage-RB readout."""
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.max_depth)

                with for_each_(m, self.circuit_depths):
                    assign(saved_gate, sequence_list[m])
                    assign(sequence_list[m], inv_gate_list[m - 1])

                    with for_(n, 0, n < self.shots, n + 1):
                        wait(resonator_dump_time*u.ns, self.element)
                        reset_frame(qubit)
                        align()
                        play_sequence(qubit, sequence_list, m)

                        # SS Readout — save raw I/Q for offline 3-state classification
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        assign(I_reset, I)
                        align(qubit, resonator)

                        # Active reset (uses 2-state thresholds, same as the 2-state program)
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                    assign(sequence_list[m], saved_gate)
                save(k, k_stream)
            with stream_processing():
                k_stream.save("iteration")
                I_stream.buffer(self.shots).buffer(len(self.circuit_depths)).save_all("I")
                Q_stream.buffer(self.shots).buffer(len(self.circuit_depths)).save_all("Q")
        return prog

    def make_sim_prog(self):
        """Bare RB sequence for simulation: no thermalization, measurement, or active reset."""
        qubit = self.meta_dict['q_name']
        depth = self.circuit_depths[-1]
        with program() as prog:
            seq_stream = declare_stream()
            i = declare(int)
            sequence_list, _ = generate_sequence(self.seed, depth) # * 2)
            with for_(i, 0, i <= depth, i + 1):
                save(sequence_list[i], seq_stream)
            reset_frame(qubit)
            # with strict_timing_():
            play_sequence(qubit, sequence_list, depth)
            with stream_processing():
                seq_stream.save_all("sim_sequence")
        return prog

    def print_sim_sequence(self, job):
        gate_names = [
            'wait',                 # 0
            'X90, X90',             # 1
            'Y90, Y90',             # 2
            'Y90, Y90, X90, X90',   # 3
            'X90, Y90',             # 4
            'X90, mY90',            # 5
            'mX90, Y90',            # 6
            'mX90, mY90',           # 7
            'Y90, X90',             # 8
            'Y90, mX90',            # 9
            'mY90, X90',            # 10
            'mY90, mX90',           # 11
            'X90',                  # 12
            'mX90',                 # 13
            'Y90',                  # 14
            'mY90',                 # 15
            'mX90, Y90, X90',       # 16
            'mX90, mY90, X90',      # 17
            'X90, X90, Y90',        # 18
            'X90, X90, mY90',       # 19
            'Y90, Y90, X90',        # 20
            'Y90, Y90, mX90',       # 21
            'X90, Y90, X90',        # 22
            'mX90, Y90, mX90',      # 23
        ]
        handle = job.result_handles.get("sim_sequence")
        handle.wait_for_all_values()
        sequence = handle.fetch_all()['value'].astype(int)
        depth = self.circuit_depths[-1]
        print(f"\nRB sequence (depth={depth}, seed={self.seed}):")
        for i, gate in enumerate(sequence):
            suffix = "  <- recovery" if i == len(sequence) - 1 else ""
            print(f"  [{i:3d}]  Clifford {gate:2d}:  {gate_names[gate]}{suffix}")

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

    def analyze(self, job, qm):
        if self.use_3state:
            return self.analyze_3state(job, qm)
        results = fetching_tool(job, data_list=["ss", "iteration"], mode="live")

        def fitRB(x, a, p, b):
            return a * p ** x + b

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line, = ax.plot([], [], 'o', label='data')
        fit_line, = ax.plot([], [], '--', color='C0', alpha=0.7, label='fit')
        ax.set_xlabel('Num Cliffords')
        ax.set_ylabel('Return probability')
        ax.set_xscale('log')
        ax.set_xlim(self.circuit_depths[0], self.circuit_depths[-1])
        ax.set_ylim(0.4, 1)
        ax.grid(alpha=0.3, which='both')
        ax.legend()
        fig.suptitle(f"RB on {self.element}")
        interrupt_on_close(fig, job)

        ss_all = None
        return_prob = None
        while results.is_processing():
            ss_all, iteration = results.fetch_all()
            if ss_all.shape[0] == 0:
                continue
            print(f"Circuit: {iteration}/{self.num_circuits}", end="\r", flush=True)
            # ss_all shape: [circuits_elapsed, n_depths] — each entry is shot-averaged excited fraction
            return_prob = 1 - np.mean(ss_all, axis=0)
            line.set_data(self.circuit_depths, return_prob)
            try:
                popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p, p_err = popt[1], np.sqrt(pcov[1][1])
                gate_fidel = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                fit_x = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 200)
                fit_line.set_data(fit_x, fitRB(fit_x, *popt))
                ax.set_title(f'F = {gate_fidel:.8f} ± {gate_fidel_err:.8f}')
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view(scaley=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        # Final fit on completed data
        if return_prob is not None:
            try:
                popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p = popt[1]
                p_err = np.sqrt(pcov[1][1])
                clif_fidel = 1 - 0.5 * (1 - p)
                gate_fidel = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                print(f'1Q RB fidelity: {gate_fidel:.6f} ± {gate_fidel_err:.6f}')
                self.data.update({
                    'gate_fidelity': gate_fidel,
                    'gate_fidelity_error': gate_fidel_err,
                    'clifford_fidelity': clif_fidel,
                    'fit_popt': popt, 'fit_pcov': pcov
                })
            except Exception:
                print("RB Fit Failed.")

        self.data.update({'return_prob': return_prob})
        if ss_all is not None:
            # Per-circuit, per-depth shot-averaged excited fraction, before averaging over
            # circuits into return_prob. Kept as the closest available "raw data".
            self.data.update({'raw_data': ss_all})

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

        return return_prob

    @staticmethod
    def _classify_3state(I, Q, centroids):
        """Nearest-centroid state assignment (0/1/2). Vectorized equivalent of
        ss_helpers.find_state_1q_nL with the SingleShot1Q3Level centroids."""
        I = np.asarray(I)[..., None]
        Q = np.asarray(Q)[..., None]
        d2 = (I - centroids[:, 0]) ** 2 + (Q - centroids[:, 1]) ** 2
        return np.argmin(d2, axis=-1)

    def analyze_3state(self, job, qm):
        """3-state readout analysis. Classifies each shot into |0>/|1>/|2> and forms the return
        probability renormalized over the computational subspace, return_prob = N0 / (N0 + N1),
        discarding |2> shots so leakage is not misassigned into |1>."""
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

        def fitRB(x, a, p, b):
            return a * p ** x + b

        depths = np.asarray(self.circuit_depths)
        centroids = np.asarray(self.calib_params_3state, dtype=float)  # (3, 2)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line, = ax.plot([], [], 'o', label='return prob (renorm. {0,1})')
        fit_line, = ax.plot([], [], '--', color='C0', alpha=0.7, label='fit')
        leak_line, = ax.plot([], [], 's', color='C3', alpha=0.6, label='|2⟩ population')
        ax.set_xlabel('Num Cliffords')
        ax.set_ylabel('Probability')
        ax.set_xscale('log')
        ax.set_xlim(depths[0], depths[-1])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, which='both')
        ax.legend()
        fig.suptitle(f"RB (3-state readout) on {self.element}")
        interrupt_on_close(fig, job)

        return_prob = None
        leak_pop = None
        while results.is_processing():
            I, Q, iteration = results.fetch_all()
            if I is None or Q is None:
                continue
            I = np.asarray(I)
            Q = np.asarray(Q)
            if I.shape != Q.shape or I.size == 0:
                continue
            print(f"Circuit: {iteration}/{self.num_circuits}", end="\r", flush=True)
            # I, Q shape: [circuits_elapsed, n_depths, shots]
            ss = self._classify_3state(I, Q, centroids)
            N0 = np.sum(ss == 0, axis=(0, 2)).astype(float)
            N1 = np.sum(ss == 1, axis=(0, 2)).astype(float)
            N2 = np.sum(ss == 2, axis=(0, 2)).astype(float)
            denom01 = N0 + N1
            return_prob = np.divide(N0, denom01, out=np.full_like(N0, np.nan), where=denom01 > 0)
            leak_pop = N2 / np.maximum(N0 + N1 + N2, 1.0)
            line.set_data(depths, return_prob)
            leak_line.set_data(depths, leak_pop)
            try:
                mask = np.isfinite(return_prob)
                popt, pcov = curve_fit(fitRB, depths[mask], return_prob[mask],
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p, p_err = popt[1], np.sqrt(pcov[1][1])
                gate_fidel = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                fit_x = np.linspace(depths[0], depths[-1], 200)
                fit_line.set_data(fit_x, fitRB(fit_x, *popt))
                ax.set_title(f'F = {gate_fidel:.8f} ± {gate_fidel_err:.8f}')
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view(scaley=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        if return_prob is not None:
            try:
                mask = np.isfinite(return_prob)
                popt, pcov = curve_fit(fitRB, depths[mask], return_prob[mask],
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p = popt[1]
                p_err = np.sqrt(pcov[1][1])
                clif_fidel = 1 - 0.5 * (1 - p)
                gate_fidel = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                print(f'1Q RB fidelity (3-state, renorm {{0,1}}): {gate_fidel:.6f} ± {gate_fidel_err:.6f}')
                self.data.update({
                    'gate_fidelity': gate_fidel,
                    'gate_fidelity_error': gate_fidel_err,
                    'clifford_fidelity': clif_fidel,
                    'fit_popt': popt, 'fit_pcov': pcov,
                })
            except Exception:
                print("RB Fit Failed.")

        self.data.update({
            'return_prob': return_prob,
            'leakage_population': leak_pop,
            'readout_mode': '3state_renorm_01',
        })

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

        return return_prob

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

#
def play_sequence(element, sequence_list, depth):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(4, element) # 4 clock cycles = 16 ns
            with case_(1):
                play("X90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)
            with case_(2):
                play("Y90", element)
                play("Y90", element)
                #frame_rotation_2pi(0, element)
            with case_(3):
                play("Y90", element)
                play("Y90", element)
                play("X90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(4):
                play("X90", element)
                play("Y90", element)
                #frame_rotation_2pi(0, element)

            with case_(5):
                play("X90", element)
                play("mY90", element)
                #frame_rotation_2pi(0, element)

            with case_(6):
                play("mX90", element)
                play("Y90", element)
                #frame_rotation_2pi(0, element)

            with case_(7):
                play("mX90", element)
                play("mY90", element)
                #frame_rotation_2pi(0, element)

            with case_(8):
                play("Y90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(9):
                play("Y90", element)
                play("mX90", element)
                #frame_rotation_2pi(0, element)

            with case_(10):
                play("mY90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(11):
                play("mY90", element)
                play("mX90", element)
                #frame_rotation_2pi(0, element)

            with case_(12):
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(13):
                play("mX90", element)
                #frame_rotation_2pi(0, element)

            with case_(14):
                play("Y90", element)
                #frame_rotation_2pi(0, element)

            with case_(15):
                play("mY90", element)
                #frame_rotation_2pi(0, element)

            with case_(16):
                play("mX90", element)
                play("Y90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(17):
                play("mX90", element)
                play("mY90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(18):
                play("X90", element)
                play("X90", element)
                play("Y90", element)
                #frame_rotation_2pi(0, element)

            with case_(19):
                play("X90", element)
                play("X90", element)
                play("mY90", element)
                #frame_rotation_2pi(0, element)

            with case_(20):
                play("Y90", element)
                play("Y90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(21):
                play("Y90", element)
                play("Y90", element)
                play("mX90", element)
                #frame_rotation_2pi(0, element)

            with case_(22):
                play("X90", element)
                play("Y90", element)
                play("X90", element)
                #frame_rotation_2pi(0, element)

            with case_(23):
                play("mX90", element)
                play("Y90", element)
                play("mX90", element)
                #frame_rotation_2pi(0, element)
# # #
# def play_sequence(element, sequence_list, depth, meta_dict):
#     """
#     Helper Function to play a RB sequence
#     """
#     i = declare(int)
#     with for_(i, 0, i <= depth, i + 1):
#         with switch_(sequence_list[i], unsafe=True):
#             with case_(0):
#                 wait(4, element)
#             with case_(1):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#             with case_(2):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#             with case_(3):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])


#             with case_(4):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])

#             with case_(5):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "mY90", phase=meta_dict["Y90"]['phase'])
#             with case_(6):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])

#             with case_(7):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "mY90",

#                            phase=meta_dict["Y90"]['phase'])
#             with case_(8):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(9):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])

#             with case_(10):
#                 play_pulse(element, "mY90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(11):
#                 play_pulse(element, "mY90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])

#             with case_(12):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(13):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])

#             with case_(14):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])

#             with case_(15):
#                play_pulse(element, "mY90",  phase=meta_dict["Y90"]['phase'])

#             with case_(16):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(17):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "mY90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(18):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])

#             with case_(19):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "mY90",  phase=meta_dict["Y90"]['phase'])

#             with case_(20):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(21):
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])

#             with case_(22):
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "X90",  phase=meta_dict["X90"]['phase'])

#             with case_(23):
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])
#                 play_pulse(element, "Y90",  phase=meta_dict["Y90"]['phase'])
#                 play_pulse(element, "mX90",  phase=meta_dict["X90"]['phase'])

if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    QUBIT = 'q6'
    CIRCUIT_DEPTHS = [4]
    NUM_CIRCUITS = 1
    SHOTS = 1
    SIM_DURATION = 5_000//4  # clock cycles (4 ns each)

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {"threshold": 0, "ground_threshold": 0}

    exp = RB1Q(
        QUBIT, CIRCUIT_DEPTHS, NUM_CIRCUITS, SHOTS, calib_params,
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix='RB1Q_sim',
        config=config,
        simulate=True,
        sim_duration=SIM_DURATION,
    )