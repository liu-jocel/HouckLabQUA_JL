from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
import random as rd
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RBPurity1Q(ExperimentClass):
    """
    Purity randomized benchmarking.

    Runs random Clifford sequences without the recovery gate and measures the
    Bloch vector (X, Y, Z) of the output state using single-shot readout with
    tomographic pre-rotations:
        Z — no pre-rotation
        Y — X90 before readout
        X — mY90 before readout

    Purity is tr[rho^2] = (1/2)(<X>^2 + <Y>^2 + <Z>^2), fit to a*p^m + b.
    Gate fidelity extracted with the same formula as standard RB.
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params,
                 outer_folder='', path='', suffix='data', meta_dict=None, config=None,
                 simulate=False, sim_duration=10_000):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)

        self.element = element
        self.config = config
        self.max_depth = max(circuit_depths) + 1
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1_000_000)

        self.data = {
            "meta_dict": meta_dict,
            "circuit_depths": circuit_depths,
            "num_circuits": num_circuits,
            "shots": shots,
            "seed": self.seed,
        }

        if simulate:
            _, _, sim_job = self.simulate(self.make_sim_prog(), self.config, duration=sim_duration)
            self.print_sim_sequence(sim_job)
        else:
            self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
            self.analyze(self.job, self.qm)
            self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            k = declare(int)
            basis = declare(int)
            m = declare(int)
            n = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            Z_stream = declare_stream()
            Y_stream = declare_stream()
            X_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list = _generate_sequence(self.seed, self.max_depth)

                # basis: 0=Z (no pre-rotation), 1=Y (X90), 2=X (mY90)
                with for_each_(basis, [0, 1, 2]):
                    with for_each_(m, self.circuit_depths):
                        with for_(n, 0, n < self.shots, n + 1):
                            wait(resonator_dump_time * u.ns, self.element)
                            reset_frame(qubit)
                            align()
                            _play_sequence(qubit, sequence_list, m, self.meta_dict)
                            align(qubit, resonator)
                            with switch_(basis, unsafe=True):
                                with case_(0):
                                    wait(4, qubit)
                                with case_(1):
                                    play('X90', qubit)
                                with case_(2):
                                    play('mY90', qubit)
                            align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            with if_(I > ss_thresh):
                                assign(state, 1)
                            with else_():
                                assign(state, 0)
                            with switch_(basis, unsafe=True):
                                with case_(0):
                                    save(state, Z_stream)
                                with case_(1):
                                    save(state, Y_stream)
                                with case_(2):
                                    save(state, X_stream)
                            align(qubit, resonator)
                            with while_(I_reset > ground_confidence_threshold):
                                with if_(I_reset > ss_thresh):
                                    play('X180', qubit)
                                    align(qubit, resonator)
                                I, Q = measure_1q(self.meta_dict, I, Q)
                                assign(I_reset, I)

                save(k, k_stream)

            with stream_processing():
                k_stream.save("iteration")
                Z_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depths)).save_all("Z_ss")
                Y_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depths)).save_all("Y_ss")
                X_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depths)).save_all("X_ss")

        return prog

    def make_sim_prog(self):
        """Bare purity RB sequence for simulation — no thermalization, measurement, or reset."""
        qubit = self.meta_dict['q_name']
        depth = self.circuit_depths[-1]
        with program() as prog:
            seq_stream = declare_stream()
            i = declare(int)
            sequence_list = _generate_sequence(self.seed, self.max_depth)
            with for_(i, 0, i <= depth, i + 1):
                save(sequence_list[i], seq_stream)
            reset_frame(qubit)
            _play_sequence(qubit, sequence_list, depth, self.meta_dict)
            with stream_processing():
                seq_stream.save_all("sim_sequence")
        return prog

    def print_sim_sequence(self, job):
        gate_names = [
            'wait',
            'X90, X90',
            'Y90, Y90',
            'Y90, Y90, X90, X90',
            'X90, Y90',
            'X90, mY90',
            'mX90, Y90',
            'mX90, mY90',
            'Y90, X90',
            'Y90, mX90',
            'mY90, X90',
            'mY90, mX90',
            'X90',
            'mX90',
            'Y90',
            'mY90',
            'mX90, Y90, X90',
            'mX90, mY90, X90',
            'X90, X90, Y90',
            'X90, X90, mY90',
            'Y90, Y90, X90',
            'Y90, Y90, mX90',
            'X90, Y90, X90',
            'mX90, Y90, mX90',
        ]
        handle = job.result_handles.get("sim_sequence")
        handle.wait_for_all_values()
        sequence = handle.fetch_all()['value'].astype(int)
        depth = self.circuit_depths[-1]
        print(f"\nPurity RB sequence (depth={depth}, seed={self.seed}):")
        for i, gate in enumerate(sequence):
            print(f"  [{i:3d}]  Clifford {gate:2d}:  {gate_names[gate]}")

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["Z_ss", "Y_ss", "X_ss", "iteration"], mode="live")

        def fit_purity(x, a, p, b):
            return a * p ** x + b

        fig, (ax_comp, ax_pur) = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)

        line_X, = ax_comp.plot([], [], 'o-', label='⟨X⟩', color='C0', markersize=4)
        line_Y, = ax_comp.plot([], [], 'o-', label='⟨Y⟩', color='C1', markersize=4)
        line_Z, = ax_comp.plot([], [], 'o-', label='⟨Z⟩', color='C2', markersize=4)
        ax_comp.set_xscale('log')
        ax_comp.set_xlim(self.circuit_depths[0], self.circuit_depths[-1])
        ax_comp.set_ylim(-1.05, 1.05)
        ax_comp.set_xlabel('Num Cliffords')
        ax_comp.set_ylabel('Bloch component')
        ax_comp.legend()
        ax_comp.grid(alpha=0.3, which='both')
        ax_comp.set_title(f"Bloch components — {self.element}")

        line_pur, = ax_pur.plot([], [], 'o', label='purity', color='C3', markersize=4)
        fit_line, = ax_pur.plot([], [], '--', color='C3', alpha=0.7, label='fit')
        ax_pur.set_xscale('log')
        ax_pur.set_xlim(self.circuit_depths[0], self.circuit_depths[-1])
        ax_pur.set_ylim(-0.05, 0.55)
        ax_pur.set_xlabel('Num Cliffords')
        ax_pur.set_ylabel(r'Purity  $(\langle X\rangle^2+\langle Y\rangle^2+\langle Z\rangle^2)/2$')
        ax_pur.legend()
        ax_pur.grid(alpha=0.3, which='both')
        ax_pur.set_title("Purity RB")

        fig.suptitle(f"Purity RB — {self.element}")
        interrupt_on_close(fig, job)

        exp_X = exp_Y = exp_Z = purity = None

        while results.is_processing():
            Z_ss, Y_ss, X_ss, iteration = results.fetch_all()
            if Z_ss.shape[0] == 0:
                continue
            print(f"Circuit: {int(iteration)}/{self.num_circuits}", end="\r", flush=True)

            exp_X = 1 - 2 * np.mean(X_ss, axis=0)
            exp_Y = 1 - 2 * np.mean(Y_ss, axis=0)
            exp_Z = 1 - 2 * np.mean(Z_ss, axis=0)
            purity = 0.5 * (exp_X**2 + exp_Y**2 + exp_Z**2)

            line_X.set_data(self.circuit_depths, exp_X)
            line_Y.set_data(self.circuit_depths, exp_Y)
            line_Z.set_data(self.circuit_depths, exp_Z)
            line_pur.set_data(self.circuit_depths, purity)

            try:
                popt, pcov = curve_fit(fit_purity, self.circuit_depths, purity,
                                       [0.4, 0.99, 0.0], maxfev=5000)
                p_live = popt[1]
                gate_fidel_live = 1 - 0.5 * (1 - p_live) / 2.2083
                fit_x = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 200)
                fit_line.set_data(fit_x, fit_purity(fit_x, *popt))
                ax_pur.set_title(
                    f"Purity RB  p = {p_live:.6f}  F = {gate_fidel_live:.6f}"
                )
            except Exception:
                pass

            ax_comp.relim()
            ax_comp.autoscale_view(scaley=False)
            ax_pur.relim()
            ax_pur.autoscale_view(scaley=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")

        # if purity is not None:
        #     try:
        #         popt, pcov = curve_fit(fit_purity, self.circuit_depths, purity,
        #                                [0.4, 0.99, 0.0], maxfev=5000)
        #         p, p_err = popt[1], np.sqrt(pcov[1][1])
        #         clif_fidel = 1 - 0.5 * (1 - p)
        #         gate_fidel = 1 - 0.5 * (1 - p) / 2.2083
        #         gate_fidel_err = p_err * 0.5 / 2.2083
        #         print(f"Purity RB: p = {p:.6f} ± {p_err:.6f}  "
        #               f"F_gate = {gate_fidel:.6f} ± {gate_fidel_err:.6f}")
        #         self.data.update({
        #             'unitarity': p,
        #             'unitarity_error': p_err,
        #             'clifford_fidelity': clif_fidel,
        #             'gate_fidelity': gate_fidel,
        #             'gate_fidelity_error': gate_fidel_err,
        #             'fit_popt': popt,
        #             'fit_pcov': pcov,
        #         })
        #     except Exception:
        #         print("Purity RB fit failed.")

        self.data.update({
            'exp_X': exp_X,
            'exp_Y': exp_Y,
            'exp_Z': exp_Z,
            'purity': purity,
        })

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()


def _generate_sequence(seed, longest_circuit):
    cayley = declare(int, value=c1_table.flatten().tolist())
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=longest_circuit + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < longest_circuit, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)

    return sequence


def _play_sequence(element, sequence_list, depth, meta_dict):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(4, element)
            with case_(1):
                play("X90", element)
                play("X90", element)
            with case_(2):
                play("Y90", element)
                play("Y90", element)
            with case_(3):
                play("Y90", element)
                play("Y90", element)
                play("X90", element)
                play("X90", element)
            with case_(4):
                play("X90", element)
                play("Y90", element)
            with case_(5):
                play("X90", element)
                play("mY90", element)
            with case_(6):
                play("mX90", element)
                play("Y90", element)
            with case_(7):
                play("mX90", element)
                play("mY90", element)
            with case_(8):
                play("Y90", element)
                play("X90", element)
            with case_(9):
                play("Y90", element)
                play("mX90", element)
            with case_(10):
                play("mY90", element)
                play("X90", element)
            with case_(11):
                play("mY90", element)
                play("mX90", element)
            with case_(12):
                play("X90", element)
            with case_(13):
                play("mX90", element)
            with case_(14):
                play("Y90", element)
            with case_(15):
                play("mY90", element)
            with case_(16):
                play("mX90", element)
                play("Y90", element)
                play("X90", element)
            with case_(17):
                play("mX90", element)
                play("mY90", element)
                play("X90", element)
            with case_(18):
                play("X90", element)
                play("X90", element)
                play("Y90", element)
            with case_(19):
                play("X90", element)
                play("X90", element)
                play("mY90", element)
            with case_(20):
                play("Y90", element)
                play("Y90", element)
                play("X90", element)
            with case_(21):
                play("Y90", element)
                play("Y90", element)
                play("mX90", element)
            with case_(22):
                play("X90", element)
                play("Y90", element)
                play("X90", element)
            with case_(23):
                play("mX90", element)
                play("Y90", element)
                play("mX90", element)


if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    QUBIT = 'q1'
    CIRCUIT_DEPTHS = [4]
    NUM_CIRCUITS = 1
    SHOTS = 1
    SIM_DURATION = 5_000 // 4  # clock cycles (4 ns each)

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {"threshold": 0, "ground_threshold": 0}

    exp = RBPurity1Q(
        QUBIT, CIRCUIT_DEPTHS, NUM_CIRCUITS, SHOTS, calib_params,
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix='RBPurity_sim',
        config=config,
        simulate=True,
        sim_duration=SIM_DURATION,
    )
