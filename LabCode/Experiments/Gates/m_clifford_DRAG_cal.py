from qm.qua import *
import numpy as np
from datetime import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.ss_helpers import find_state_1q_nL
from LabCode.Helpers.qua_helpers import measure_1q
import random as rd
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class CliffordDRAGCal(ExperimentClass):
    """
    Single-depth leakage RB DRAG calibration.

    For each alpha value plays num_circuits random Clifford sequences of
    circuit_depth gates (closed with an inverse gate), with pre-measurement
    postselection on |0>. Plots the postselected |2> population vs alpha and
    marks the minimum.
    """

    def __init__(self, element, alpha_vec, circuit_depth, num_circuits, shots, calib_params,
                 outer_folder='', path='', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)
        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.alpha_vec = np.asarray(alpha_vec, dtype=float)
        self.circuit_depth = circuit_depth
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1000000)

        self.data = {
            "meta_dict": meta_dict,
            'config': config,
            'alpha_vec': self.alpha_vec,
            "circuit_depth": circuit_depth,
            "num_circuits": num_circuits,
            "shots": shots,
            'seed': self.seed,
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']

        with program() as prog:
            k = declare(int)
            n = declare(int)
            a = declare(fixed)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_ps_stream = declare_stream()
            Q_ps_stream = declare_stream()
            I_stream = declare_stream()
            Q_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.circuit_depth)

                # Close the circuit: replace gate at circuit_depth with its inverse
                assign(saved_gate, sequence_list[self.circuit_depth])
                assign(sequence_list[self.circuit_depth], inv_gate_list[self.circuit_depth - 1])

                with for_each_(a, self.alpha_vec):
                    with for_(n, 0, n < self.shots, n + 1):
                        # Pre-measurement for postselection
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_ps_stream)
                        save(Q, Q_ps_stream)
                        wait(resonator_dump_time * u.ns)

                        # RB sequence (circuit_depth random gates + 1 inverse)
                        align()
                        play_sequence(self.element, sequence_list, self.circuit_depth, a)

                        # Result measurement
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(resonator_dump_time * u.ns)

                assign(sequence_list[self.circuit_depth], saved_gate)
                save(k, k_stream)

            with stream_processing():
                k_stream.save("iteration")
                # Shape after save_all: (n_circuits, n_alpha, n_shots)
                I_ps_stream.buffer(self.shots).buffer(len(self.alpha_vec)).save_all("I_ps")
                Q_ps_stream.buffer(self.shots).buffer(len(self.alpha_vec)).save_all("Q_ps")
                I_stream.buffer(self.shots).buffer(len(self.alpha_vec)).save_all("I")
                Q_stream.buffer(self.shots).buffer(len(self.alpha_vec)).save_all("Q")

        return prog

    def analyze(self, job, qm):
        matplotlib.use("QtAgg")
        n_alpha = len(self.alpha_vec)
        data_list = ["I_ps", "Q_ps", "I", "Q", "iteration"]
        results = fetching_tool(job, data_list=data_list, mode="live")

        fig, ax = plt.subplots()
        pop2_line, = ax.plot(self.alpha_vec, np.full(n_alpha, np.nan), '.',
                             color='C0', ms=6, label='|2⟩')
        min_marker, = ax.plot([], [], 'o', color='red', ms=10, zorder=3, label='min')
        ax.set_xlabel("DRAG alpha")
        ax.set_ylabel("|2⟩ population (postselected)")
        ax.set_ylim(0, 1)
        ax.set_xlim(self.alpha_vec.min(), self.alpha_vec.max())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.suptitle(f"{self.element} Clifford DRAG Cal — depth {self.circuit_depth}")
        interrupt_on_close(fig, job)

        # Incremental per-alpha accumulators: list of (res_state_1d, valid_1d) chunks
        ss_buffers = [[] for _ in range(n_alpha)]
        last_processed = 0
        draw_period = 0.2
        last_draw = time.time()
        p2 = np.full(n_alpha, np.nan)

        while results.is_processing():
            I_ps, Q_ps, I, Q, iteration = results.fetch_all()
            print(f"\r{int(iteration)}/{self.num_circuits} circuits", end="", flush=True)

            # Align streams: truncate to the shortest common first dimension
            if not (I_ps.shape == Q_ps.shape == I.shape == Q.shape):
                n_min = min(I_ps.shape[0], Q_ps.shape[0], I.shape[0], Q.shape[0])
                if n_min == 0:
                    continue
                I_ps, Q_ps, I, Q = I_ps[:n_min], Q_ps[:n_min], I[:n_min], Q[:n_min]

            if I.ndim != 3 or I.shape[1:] != (n_alpha, self.shots):
                continue

            n_circuits_so_far = I.shape[0]
            if n_circuits_so_far > last_processed:
                I_ps_new = I_ps[last_processed:n_circuits_so_far]
                Q_ps_new = Q_ps[last_processed:n_circuits_so_far]
                I_new    = I   [last_processed:n_circuits_so_far]
                Q_new    = Q   [last_processed:n_circuits_so_far]

                for ai in range(n_alpha):
                    # Flatten (new_circuits, shots) → (1, new_circuits*shots)
                    I_ps_a = I_ps_new[:, ai, :].reshape(1, -1)
                    Q_ps_a = Q_ps_new[:, ai, :].reshape(1, -1)
                    I_a    = I_new   [:, ai, :].reshape(1, -1)
                    Q_a    = Q_new   [:, ai, :].reshape(1, -1)

                    ps_state  = find_state_1q_nL(I_ps_a, Q_ps_a, self.calib_params)
                    res_state = find_state_1q_nL(I_a,    Q_a,    self.calib_params)
                    valid = (ps_state == 0)
                    ss_buffers[ai].append((res_state.ravel(), valid.ravel()))

                last_processed = n_circuits_so_far

            now = time.time()
            if now - last_draw > draw_period:
                for ai in range(n_alpha):
                    if not ss_buffers[ai]:
                        continue
                    ss_all    = np.concatenate([c[0] for c in ss_buffers[ai]])
                    valid_all = np.concatenate([c[1] for c in ss_buffers[ai]])
                    n_kept = int(valid_all.sum())
                    if n_kept > 0:
                        p2[ai] = float(((ss_all == 2) & valid_all).sum()) / n_kept

                pop2_line.set_ydata(p2)
                if np.any(np.isfinite(p2)):
                    best_idx = int(np.nanargmin(p2))
                    min_marker.set_data([self.alpha_vec[best_idx]], [p2[best_idx]])
                    ax.set_title(f"Best alpha: {self.alpha_vec[best_idx]:.4f}  "
                                 f"|2⟩ = {p2[best_idx]:.4f}")

                plt.pause(0.05)
                last_draw = now

        print("")
        qm.close()

        # Final populations
        for ai in range(n_alpha):
            if not ss_buffers[ai]:
                continue
            ss_all    = np.concatenate([c[0] for c in ss_buffers[ai]])
            valid_all = np.concatenate([c[1] for c in ss_buffers[ai]])
            n_kept = int(valid_all.sum())
            if n_kept > 0:
                p2[ai] = float(((ss_all == 2) & valid_all).sum()) / n_kept

        best_alpha = float(self.alpha_vec[int(np.nanargmin(p2))]) if np.any(np.isfinite(p2)) else float("nan")
        print(f"Best alpha: {best_alpha:.6f}  |2⟩ = {p2[int(np.nanargmin(p2))]:.6f}" if np.any(np.isfinite(p2)) else "No valid data")

        self.data.update({'p2': p2, 'best_alpha': best_alpha})

        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)


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


def play_sequence(element, sequence_list, depth, a):
    """
    Helper Function to play a RB sequence with DRAG alpha scaling.
    X pulses: amp(1, 0, 0, a) — Q channel scaled by a.
    Y pulses: amp(a, 0, 0, 1) — I channel scaled by a.
    """
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(4, element)
            with case_(1):
                play("X90"*amp(1, 0, 0, a), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(2):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("Y90"*amp(a, 0, 0, 1), element)
            with case_(3):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(4):
                play("X90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
            with case_(5):
                play("X90"*amp(1, 0, 0, a), element)
                play("mY90"*amp(a, 0, 0, 1), element)
            with case_(6):
                play("mX90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
            with case_(7):
                play("mX90"*amp(1, 0, 0, a), element)
                play("mY90"*amp(a, 0, 0, 1), element)
            with case_(8):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(9):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("mX90"*amp(1, 0, 0, a), element)
            with case_(10):
                play("mY90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(11):
                play("mY90"*amp(a, 0, 0, 1), element)
                play("mX90"*amp(1, 0, 0, a), element)
            with case_(12):
                play("X90"*amp(1, 0, 0, a), element)
            with case_(13):
                play("mX90"*amp(1, 0, 0, a), element)
            with case_(14):
                play("Y90"*amp(a, 0, 0, 1), element)
            with case_(15):
                play("mY90"*amp(a, 0, 0, 1), element)
            with case_(16):
                play("mX90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(17):
                play("mX90"*amp(1, 0, 0, a), element)
                play("mY90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(18):
                play("X90"*amp(1, 0, 0, a), element)
                play("X90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
            with case_(19):
                play("X90"*amp(1, 0, 0, a), element)
                play("X90"*amp(1, 0, 0, a), element)
                play("mY90"*amp(a, 0, 0, 1), element)
            with case_(20):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(21):
                play("Y90"*amp(a, 0, 0, 1), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("mX90"*amp(1, 0, 0, a), element)
            with case_(22):
                play("X90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("X90"*amp(1, 0, 0, a), element)
            with case_(23):
                play("mX90"*amp(1, 0, 0, a), element)
                play("Y90"*amp(a, 0, 0, 1), element)
                play("mX90"*amp(1, 0, 0, a), element)
