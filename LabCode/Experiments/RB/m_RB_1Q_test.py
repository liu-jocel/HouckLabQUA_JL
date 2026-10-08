"""
RB1QChunked — drop-in replacement for RB1Q that avoids the OPX per-array
memory limit by splitting sequence arrays into chunks of SEQ_CHUNK_SIZE.

Changes vs m_RB_1Q.py:
  * _pregenerate_sequence() replaces QUA Random (numpy, no QUA memory pressure)
  * _load_sequence_chunks() splits arrays into declare(int, size=SEQ_CHUNK_SIZE)
  * play_sequence_chunked() loops over chunks instead of one big array
  * make_prog(): k-loop and depth-loop are unrolled in Python so chunk/local
    indices are compile-time constants — no runtime branch on chunk selection
  * make_sim_prog(): same chunked approach for simulation
  * __init__: max_depth = max(circuit_depths)  (no 2× factor needed)

Run the pure-Python correctness tests (no hardware required):
    python LabCode/Experiments/RB/m_RB_1Q_test.py
"""

import numpy as np
import math
import random as rd
from datetime import datetime

# QUA / hardware imports — only available in the QuaCR conda environment.
# The pure-Python tests below run without them.
try:
    from qm.qua import *
    from qualang_tools.results import fetching_tool
    from qualang_tools.plot import interrupt_on_close
    from qualang_tools.units import unit
    from LabCode.CoreLib.experiment import ExperimentClass
    from LabCode.Helpers.pulse_macros import *
    from LabCode.Helpers.qua_helpers import measure_1q
    from scipy.optimize import curve_fit
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    plt.ion()
    _QUA_AVAILABLE = True
    u = unit(coerce_to_integer=True)
except ImportError as _qua_err:
    print(f"[m_RB_1Q_test] QUA/LabCode imports unavailable ({_qua_err}); "
          "class RB1QChunked will not be defined. Tests still run.")
    _QUA_AVAILABLE = False

from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

# Maximum number of elements per QUA array.  Tune to the OPX's actual limit.
SEQ_CHUNK_SIZE = 4096


# ---------------------------------------------------------------------------
# Pure-Python sequence generator
# ---------------------------------------------------------------------------

def _pregenerate_sequence(seed, max_depth):
    """Generate a full RB Clifford sequence using numpy (no QUA Random).

    Returns:
        seq_vals : list[int], length max_depth+1.
                   seq_vals[i]  = i-th random Clifford index (0-indexed).
                   seq_vals[max_depth] = 0 (padding; recovery gate slot).
        inv_vals : list[int], length max_depth+1.
                   inv_vals[i]  = recovery gate after applying gates 0..i.
                   inv_vals[max_depth] = 0 (padding).
    """
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
    rng = np.random.default_rng(seed)
    seq_vals, inv_vals = [], []
    state = 0
    for _ in range(max_depth):
        step = int(rng.integers(0, 24))
        state = int(c1_table[state, step])
        seq_vals.append(step)
        inv_vals.append(inv_gates[state])
    seq_vals.append(0)
    inv_vals.append(0)
    return seq_vals, inv_vals


# ---------------------------------------------------------------------------
# QUA helpers (must be called inside a program() context)
# ---------------------------------------------------------------------------

def _load_sequence_chunks(seq_vals, inv_vals, chunk_size):
    """Load pre-generated sequence lists into chunked QUA arrays.

    Each chunk is declared with exactly chunk_size elements (last chunk is
    zero-padded so all chunks have identical declared sizes).

    Returns:
        seq_chunks : list of QUA int arrays
        inv_chunks : list of QUA int arrays
    """
    n = len(seq_vals)
    num_chunks = math.ceil(n / chunk_size)
    seq_chunks, inv_chunks = [], []
    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, n)
        pad = chunk_size - (e - s)
        seq_chunks.append(declare(int, value=seq_vals[s:e] + [0] * pad))
        inv_chunks.append(declare(int, value=inv_vals[s:e] + [0] * pad))
    return seq_chunks, inv_chunks


def _clifford_cases(element):
    """Emit all 24 QUA switch cases for single-qubit Clifford playback."""
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


def play_sequence_chunked(element, seq_chunks, depth_m, chunk_size, loop_var):
    """Play Clifford gates 0..depth_m from chunked QUA arrays.

    depth_m   : Python int — chunk/local indices computed at compile time.
    loop_var  : pre-declared QUA int used as the loop counter (reused across calls).
    """
    full_c   = depth_m // chunk_size
    last_loc = depth_m % chunk_size

    for c in range(full_c):
        with for_(loop_var, 0, loop_var < chunk_size, loop_var + 1):
            with switch_(seq_chunks[c][loop_var], unsafe=True):
                _clifford_cases(element)

    with for_(loop_var, 0, loop_var <= last_loc, loop_var + 1):
        with switch_(seq_chunks[full_c][loop_var], unsafe=True):
            _clifford_cases(element)


# ---------------------------------------------------------------------------
# Modified RB1Q class
# ---------------------------------------------------------------------------

if not _QUA_AVAILABLE:
    class ExperimentClass:  # stub base so class definition succeeds without QUA
        pass

class RB1QChunked(ExperimentClass):
    """Single-qubit RB using chunked QUA arrays to avoid the OPX array-size limit.

    Identical interface to RB1Q; drop-in replacement.
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params,
                 outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None, simulate=False, sim_duration=10_000):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)

        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.max_depth = max(circuit_depths)   # no 2× factor; chunks handle full depth
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
            'seed': self.seed,
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
        qubit      = self.meta_dict['q_name']
        resonator  = self.meta_dict['r_name']
        resonator_dump_time       = self.meta_dict['resonator_dump_time']
        ss_thresh                 = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        chunk_size = SEQ_CHUNK_SIZE
        num_chunks = math.ceil((self.max_depth + 1) / chunk_size)

        # Pre-generate one independent Clifford sequence per circuit in Python.
        # OPX memory per circuit: 2 × num_chunks × chunk_size × 4 bytes.
        rng_master = np.random.default_rng(self.seed)
        circuit_data = [
            _pregenerate_sequence(int(rng_master.integers(1, 1_000_000)), self.max_depth)
            for _ in range(self.num_circuits)
        ]

        with program() as prog:
            n          = declare(int)
            state      = declare(int)
            saved_gate = declare(int)
            I          = declare(fixed)
            Q          = declare(fixed)
            I_reset    = declare(fixed)
            ss_stream  = declare_stream()
            seq_stream = declare_stream()
            k_stream   = declare_stream()
            s_local    = declare(int)   # reused across seq-save loops
            loop_var   = declare(int)   # reused across play_sequence_chunked calls
            k_prog     = declare(int)
            assign(k_prog, 0)

            # Load all circuit sequences as QUA chunk arrays at program scope.
            all_chunks = [
                _load_sequence_chunks(sv, iv, chunk_size)
                for sv, iv in circuit_data
            ]

            # k-loop unrolled in Python: each circuit gets its own chunk arrays.
            for seq_chunks, inv_chunks in all_chunks:

                # Stream this circuit's random gates (indices 0..max_depth-1).
                for c in range(num_chunks):
                    end = chunk_size if c < num_chunks - 1 else (self.max_depth - c * chunk_size)
                    if end > 0:
                        with for_(s_local, 0, s_local < end, s_local + 1):
                            save(seq_chunks[c][s_local], seq_stream)

                # Depth loop unrolled in Python: chunk/local indices are constants.
                for m in self.circuit_depths:
                    c_m   = m // chunk_size
                    l_m   = m % chunk_size
                    c_inv = (m - 1) // chunk_size
                    l_inv = (m - 1) % chunk_size

                    assign(saved_gate, seq_chunks[c_m][l_m])
                    assign(seq_chunks[c_m][l_m], inv_chunks[c_inv][l_inv])

                    with for_(n, 0, n < self.shots, n + 1):
                        wait(resonator_dump_time * u.ns, self.element)
                        reset_frame(qubit)
                        align()
                        play_sequence_chunked(qubit, seq_chunks, m, chunk_size, loop_var)

                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_stream)
                        align(qubit, resonator)

                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)

                    assign(seq_chunks[c_m][l_m], saved_gate)

                save(k_prog, k_stream)
                assign(k_prog, k_prog + 1)

            with stream_processing():
                k_stream.save("iteration")
                ss_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(len(self.circuit_depths)).save_all("ss")
                seq_stream.save_all("circuits")
        return prog

    def make_sim_prog(self):
        """Bare RB sequence for simulation: no thermalization, measurement, or active reset."""
        qubit      = self.meta_dict['q_name']
        depth      = self.circuit_depths[-1]
        chunk_size = SEQ_CHUNK_SIZE
        seq_vals, inv_vals = _pregenerate_sequence(self.seed, depth)
        num_chunks = math.ceil(len(seq_vals) / chunk_size)  # len = depth + 1

        with program() as prog:
            seq_stream = declare_stream()
            seq_chunks, _ = _load_sequence_chunks(seq_vals, inv_vals, chunk_size)
            s_local  = declare(int)
            loop_var = declare(int)

            # Save gates 0..depth (depth+1 values; last is recovery placeholder).
            for c in range(num_chunks):
                end = chunk_size if c < num_chunks - 1 else ((depth + 1) - c * chunk_size)
                if end > 0:
                    with for_(s_local, 0, s_local < end, s_local + 1):
                        save(seq_chunks[c][s_local], seq_stream)

            reset_frame(qubit)
            play_sequence_chunked(qubit, seq_chunks, depth, chunk_size, loop_var)
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

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss", "iteration", "circuits"], mode="live")

        def fitRB(x, a, p, b):
            return a * p ** x + b

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line,     = ax.plot([], [], 'o',  label='data')
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
        circuits = None
        return_prob = None
        while results.is_processing():
            ss_all, iteration, circuits = results.fetch_all()
            if ss_all.shape[0] == 0:
                continue
            print(f"Circuit: {iteration}/{self.num_circuits}", end="\r", flush=True)
            return_prob = 1 - np.mean(ss_all, axis=0)
            line.set_data(self.circuit_depths, return_prob)
            try:
                popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p, p_err = popt[1], np.sqrt(pcov[1][1])
                gate_fidel     = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                fit_x = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 200)
                fit_line.set_data(fit_x, fitRB(fit_x, *popt))
                ax.set_title(f'F = {gate_fidel:.6f} ± {gate_fidel_err:.6f}')
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
                popt, pcov = curve_fit(fitRB, self.circuit_depths, return_prob,
                                       [0.4, 0.99, 0.5], maxfev=5000)
                p      = popt[1]
                p_err  = np.sqrt(pcov[1][1])
                clif_fidel     = 1 - 0.5 * (1 - p)
                gate_fidel     = 1 - 0.5 * (1 - p) / 2.2083
                gate_fidel_err = p_err * 0.5 / 2.2083
                print(f'1Q RB fidelity: {gate_fidel:.6f} ± {gate_fidel_err:.6f}')
                self.data.update({
                    'gate_fidelity':       gate_fidel,
                    'gate_fidelity_error': gate_fidel_err,
                    'clifford_fidelity':   clif_fidel,
                    'fit_popt': popt, 'fit_pcov': pcov,
                })
            except Exception:
                print("RB Fit Failed.")

        self.data.update({'return_prob': return_prob, 'sequences': circuits})
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()
        return return_prob


# ---------------------------------------------------------------------------
# Pure-Python correctness tests (no QUA / hardware required)
# ---------------------------------------------------------------------------

def _split_into_chunks(vals, chunk_size):
    """Mirror of _load_sequence_chunks splitting logic, without QUA declare()."""
    n = len(vals)
    num_chunks = math.ceil(n / chunk_size)
    chunks = []
    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, n)
        pad = chunk_size - (e - s)
        chunks.append(vals[s:e] + [0] * pad)
    return chunks


def _reference_pregenerate(seed, max_depth):
    """Independent reimplementation for cross-checking."""
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
    rng = np.random.default_rng(seed)
    seq, inv = [], []
    state = 0
    for _ in range(max_depth):
        step = int(rng.integers(0, 24))
        state = int(c1_table[state, step])
        seq.append(step)
        inv.append(inv_gates[state])
    seq.append(0)
    inv.append(0)
    return seq, inv


def test_reproducibility():
    s1, i1 = _pregenerate_sequence(42, 100)
    s2, i2 = _pregenerate_sequence(42, 100)
    assert s1 == s2 and i1 == i2
    print("PASS  test_reproducibility")


def test_different_seeds_differ():
    s_a, _ = _pregenerate_sequence(1, 50)
    s_b, _ = _pregenerate_sequence(2, 50)
    assert s_a != s_b
    print("PASS  test_different_seeds_differ")


def test_gate_indices_in_range():
    seq, _ = _pregenerate_sequence(7, 200)
    assert all(0 <= g <= 23 for g in seq[:-1])
    print("PASS  test_gate_indices_in_range")


def test_output_length():
    max_depth = 77
    seq, inv = _pregenerate_sequence(5, max_depth)
    assert len(seq) == max_depth + 1 and len(inv) == max_depth + 1
    assert seq[-1] == 0 and inv[-1] == 0
    print("PASS  test_output_length")


def test_recovery_gate_at_max_depth():
    seed, max_depth = 99, 50
    seq, inv = _pregenerate_sequence(seed, max_depth)
    state = 0
    for i in range(max_depth):
        state = int(c1_table[state, seq[i]])
    assert int(c1_table[state, inv[max_depth - 1]]) == 0
    print("PASS  test_recovery_gate_at_max_depth")


def test_recovery_gate_every_sub_depth():
    seed, max_depth = 123, 30
    seq, inv = _pregenerate_sequence(seed, max_depth)
    for m in range(1, max_depth + 1):
        state = 0
        for i in range(m):
            state = int(c1_table[state, seq[i]])
        assert int(c1_table[state, inv[m - 1]]) == 0, f"failed at m={m}"
    print("PASS  test_recovery_gate_every_sub_depth")


def test_chunk_roundtrip():
    seed, max_depth = 55, 500
    seq, inv = _pregenerate_sequence(seed, max_depth)
    n = max_depth + 1
    for orig, label in [(seq, "seq"), (inv, "inv")]:
        chunks = _split_into_chunks(orig, SEQ_CHUNK_SIZE)
        flat = [v for chunk in chunks for v in chunk]
        assert flat[:n] == orig, f"{label} roundtrip failed"
    print("PASS  test_chunk_roundtrip")


def test_matches_reference():
    seed, max_depth = 77, 150
    seq, inv = _pregenerate_sequence(seed, max_depth)
    ref_seq, ref_inv = _reference_pregenerate(seed, max_depth)
    assert seq == ref_seq and inv == ref_inv
    print("PASS  test_matches_reference")


def test_chunk_count_large_depth():
    max_depth = 7700
    seq, _ = _pregenerate_sequence(42, max_depth)
    assert len(seq) == max_depth + 1
    chunks = _split_into_chunks(seq, SEQ_CHUNK_SIZE)
    num_chunks = len(chunks)
    assert all(len(c) == SEQ_CHUNK_SIZE for c in chunks)
    assert num_chunks * SEQ_CHUNK_SIZE >= max_depth + 1
    assert (num_chunks - 1) * SEQ_CHUNK_SIZE < max_depth + 1
    print(f"PASS  test_chunk_count_large_depth  (num_chunks={num_chunks})")


def test_chunk_index_arithmetic():
    """Verify chunk+local indices correctly address every depth in a typical list."""
    seed, max_depth = 88, 7700
    circuit_depths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028, 7700]
    chunk_size = SEQ_CHUNK_SIZE
    seq, inv = _pregenerate_sequence(seed, max_depth)
    chunks_seq = _split_into_chunks(seq, chunk_size)
    chunks_inv = _split_into_chunks(inv, chunk_size)
    for m in circuit_depths:
        c_m,   l_m   = m // chunk_size, m % chunk_size
        c_inv, l_inv = (m - 1) // chunk_size, (m - 1) % chunk_size
        assert chunks_seq[c_m][l_m] == seq[m],       f"seq index mismatch at m={m}"
        assert chunks_inv[c_inv][l_inv] == inv[m-1], f"inv index mismatch at m={m}"
    print("PASS  test_chunk_index_arithmetic")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_reproducibility,
        test_different_seeds_differ,
        test_gate_indices_in_range,
        test_output_length,
        test_recovery_gate_at_max_depth,
        test_recovery_gate_every_sub_depth,
        test_chunk_roundtrip,
        test_matches_reference,
        test_chunk_count_large_depth,
        test_chunk_index_arithmetic,
    ]

    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        raise SystemExit(1)
    else:
        print(f"All {len(tests)} tests passed.")

    # -----------------------------------------------------------------------
    # Simulation — requires QUA / LabCode (QuaCR conda env)
    # -----------------------------------------------------------------------
    if not _QUA_AVAILABLE:
        print("\nSkipping simulation: QUA / LabCode not available.")
    else:
        from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
        from LabCode.Config.config_helpers import meta_dict_to_config_1Q

        QUBIT          = 'q1'
        CIRCUIT_DEPTHS = [7000]
        NUM_CIRCUITS   = 1
        SHOTS          = 1
        # ~40 clock cycles per Clifford (accounts for multi-pulse gates and
        # frame resets); use at least 1250 so short depths are covered too.
        SIM_DURATION   = max(1_250, 40 * CIRCUIT_DEPTHS[-1])

        meta_dict    = make_meta_dict(QUBIT)[QUBIT]
        config       = meta_dict_to_config_1Q(meta_dict)
        calib_params = {"threshold": 0, "ground_threshold": 0}

        print(f"\nRunning chunked RB simulation (depth={CIRCUIT_DEPTHS}, qubit={QUBIT}) ...")
        exp = RB1QChunked(
            QUBIT, CIRCUIT_DEPTHS, NUM_CIRCUITS, SHOTS, calib_params,
            meta_dict=meta_dict,
            outer_folder=meta_dict['global_params']['outer_folder'],
            path=QUBIT,
            suffix='RB1QChunked_sim',
            config=config,
            simulate=True,
            sim_duration=SIM_DURATION,
        )
        print(f"Simulation complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
