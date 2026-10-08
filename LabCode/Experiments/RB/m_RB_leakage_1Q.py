from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.ss_helpers import find_state_1q_nL
from LabCode.Helpers.qua_helpers import measure_1q
import lmfit
from LabCode.Helpers.RB_helpers import fit_leakage_RB
from scipy.optimize import curve_fit
import random as rd
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class RB1Q_Leakage(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None, simulate=False, sim_duration=10_000, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)
        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.max_depth = max(circuit_depths)
        self.circuit_depths = np.sort(circuit_depths)[::-1]
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
        reset_time = self.meta_dict['reset_time']
        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            s = declare(int)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            k_stream = declare_stream()
            seq_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, inv_gate_list = generate_sequence(self.seed, self.max_depth)

                with for_(s, 0, s < self.max_depth, s + 1):
                    save(sequence_list[s], seq_stream)

                with for_each_(m, self.circuit_depths):
                    assign(saved_gate, sequence_list[m])
                    assign(sequence_list[m], inv_gate_list[m - 1])

                    with for_(n, 0, n < self.shots, n + 1):
                        align()
                        play_sequence(self.element, sequence_list, m)

                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(reset_time * u.ns)
                save(k, k_stream)
            with stream_processing():
                k_stream.save("iteration")
                I_stream.buffer(self.shots).buffer(len(self.circuit_depths)).save_all("I")
                Q_stream.buffer(self.shots).buffer(len(self.circuit_depths)).save_all("Q")
        return prog

    def make_sim_prog(self):
        """Bare leakage RB sequence for simulation: no measurement or reset."""
        qubit = self.meta_dict['q_name']
        depth = int(self.circuit_depths[0])
        with program() as prog:
            seq_stream = declare_stream()
            i = declare(int)
            sequence_list, _ = generate_sequence(self.seed, depth * 2)
            with for_(i, 0, i <= depth, i + 1):
                save(sequence_list[i], seq_stream)
            reset_frame(qubit)
            play_sequence(self.element, sequence_list, depth)
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
        depth = int(self.circuit_depths[0])
        print(f"\nLeakage RB sequence (depth={depth}, seed={self.seed}):")
        for i, gate in enumerate(sequence):
            suffix = "  <- recovery" if i == len(sequence) - 1 else ""
            print(f"  [{i:3d}]  Clifford {gate:2d}:  {gate_names[gate]}{suffix}")

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
        # Live Plotting
        fig, ax = plt.subplots()
        lines = [ax.plot([], [], '.', label=f'P{i}')[0] for i in range(3)]
        p11_line = ax.plot([], [], '.', color='C3', label='P11 = P0 + P1')[0]
        p11_fit_line = ax.plot([], [], '-', color='C3', alpha = 0.7)[0]
        fit_lines = [ax.plot([], [], '--', color='C' + str(i), alpha=0.7)[0] for i in
                 range(3)]  # dashed fit lines

        ax.set_xlabel("Circuit depth")
        ax.set_ylabel("Population")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.suptitle(f"Leakage RB")
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

        # --- Stream + accumulate populations in 0 , 1, and 2---
        ss_buffers = []
        populations = None
        fit_data = {}
        while results.is_processing():
            I, Q, iteration = results.fetch_all()
            if I.dtype.names:
                I = I['value']
                Q = Q['value']
            if I.shape != Q.shape:
                continue
            print(f"Circuit Depth: \r{iteration}/{self.num_circuits}", end="", flush=True)
            # I,Q have shape [circuits_elpased, depths, shots,]
            # Reshape data to be shape [depths, circuits_elapsed*shots]
            circuits_elapsed, depths, shots = I.shape
            I = np.moveaxis(I, 0, 1).reshape(depths, -1)
            Q = np.moveaxis(Q, 0, 1).reshape(depths, -1)

            # Extract single-shot states: shape (depths,  circuits_elapsed*shots)
            new_ss = find_state_1q_nL(
                np.array(I),
                np.array(Q),
                self.calib_params
            )

            ss_buffers.append(new_ss.T)

            # Final single-shot array: (total_shots, n_circuit_depths)
            if len(ss_buffers) == 1:
                ss_array = ss_buffers[0]
            else:
                ss_array = np.concatenate(ss_buffers, axis=0)

            # --- Histogram populations over shots ---
            counts = np.stack(
                [(ss_array == k).sum(axis=0) for k in (0, 1, 2)],
                axis=0
            )

            # Normalize to populations
            populations = counts / counts.sum(axis=0, keepdims=True)

            # --- Plot populations vs circuit depth ---
            for i, line in enumerate(lines):
                line.set_data(self.circuit_depths, populations[i])
            p11_line.set_data(self.circuit_depths, populations[0] + populations[1])

            # --- Fit the computational-subspace population (Wood-Gambetta M1) ---
            # p11 = p0 + p1 is themobservable used to estimate leakage and seepage.
            # Convention: L1 is leakage out of the computational subspace and
            # L2 is seepage back into it
            try:
                p11 = populations[0] + populations[1]

                def p11_model(d, A, B, lambda1):
                    return A + B * lambda1 ** d

                model = lmfit.Model(p11_model)
                params = model.make_params(A=0.95, B=0.04, lambda1=0.99)
                # Physical fit bounds: populations/amplitudes are nonnegative,
                # and lambda1 is an eigenvalue in the interval [0, 1].
                params['A'].set(min=0, max=1)
                params['B'].set(min=0, max=1)
                params['lambda1'].set(min=0, max=1)
                result = model.fit(p11, params, d=self.circuit_depths)
                p11_fit_line.set_data(self.circuit_depths, result.best_fit)
                A = result.params['A'].value
                B = result.params['B'].value
                lambda1 = result.params['lambda1'].value
                L1 = (1 - A) * (1 - lambda1)
                L2 = A * (1 - lambda1)
                L1_err = np.nan
                L2_err = np.nan
                if result.covar is not None:
                    grad_L1 = np.array([-(1 - lambda1), 0.0, -(1 - A)])
                    grad_L2 = np.array([(1 - lambda1), 0.0, -A])
                    L1_err = np.sqrt(max(0.0, grad_L1 @ result.covar @ grad_L1))
                    L2_err = np.sqrt(max(0.0, grad_L2 @ result.covar @ grad_L2))

                # The leakage-state population should rise from zero. This is
                # a consistency curve, not an additional independent fit.
                p2_fit = (L1 / (L1 + L2)) * (1 - lambda1 ** self.circuit_depths)
                fit_lines[2].set_data(self.circuit_depths, p2_fit)
                fit_data['state_2_result'] = result
                fit_data['state_2_L1'] = (L1, L1_err)
                fit_data['state_2_L2'] = (L2, L2_err)
                fit_data['state_2_lambda1'] = (lambda1, result.params['lambda1'].stderr or np.nan)
                fit_data['state_2_p11_fit'] = result.best_fit
                fit_data['state_2_p2_fit'] = p2_fit
                fit_data['state_2_L1_per_gate'] = (L1 / 2.2083, L1_err / 2.2083)
                fit_data['state_2_L2_per_gate'] = (L2 / 2.2083, L2_err / 2.2083)

                # --- Fit P0 for lambda2 and average gate fidelity ---
                # Wood-Gambetta model: P0(m) = A0 + B0*lambda1^m + C0*lambda2^m.
                # lambda1 is fixed by the P11 fit. For weak leakage, the paper
                # allows the reduced model P0(m) = A0 + C0*lambda2^m when
                # lambda2 << lambda1 and B0 << A0.
                p0 = populations[0]

                def p0_model(d, A0, B0, C0, lambda2):
                    return A0 + B0 * lambda1 ** d + C0 * lambda2 ** d

                p0_model_full = lmfit.Model(p0_model)
                p0_params = p0_model_full.make_params(A0=0.5, B0=0.0, C0=0.5, lambda2=0.99)
                # Physical bounds from the paper: populations and lambda2 are
                # bounded; B0 may have either sign because it is an amplitude.
                p0_params['A0'].set(min=0, max=A)
                p0_params['B0'].set(min=-1, max=1)
                p0_params['C0'].set(min=0, max=1)
                p0_params['lambda2'].set(min=0, max=1)
                p0_result = p0_model_full.fit(p0, p0_params, d=self.circuit_depths)
                lambda2 = p0_result.params['lambda2'].value

                b_over_a = abs(B) / A if A > 0 else np.inf
                weak_leakage = (b_over_a < 0.1) and (lambda2 < lambda1)
                if weak_leakage:
                    # Refit with the robust weak-leakage model recommended by
                    # the paper when the lambda1 component is negligible.
                    def p0_model_weak(d, A0, C0, lambda2):
                        return A0 + C0 * lambda2 ** d

                    weak_model = lmfit.Model(p0_model_weak)
                    weak_params = weak_model.make_params(A0=0.5, C0=0.5, lambda2=0.99)
                    weak_params['A0'].set(min=0, max=A)
                    weak_params['C0'].set(min=0, max=1)
                    weak_params['lambda2'].set(min=0, max=1)
                    p0_result = weak_model.fit(p0, weak_params, d=self.circuit_depths)
                    p0_fit = p0_result.best_fit
                    lambda2 = p0_result.params['lambda2'].value
                    fit_regime = 'weak'
                else:
                    p0_fit = p0_result.best_fit
                    fit_regime = 'full'

                # For a qubit (d1=2), F = [(d1-1)*lambda2 + 1 - L1] / d1.
                d1 = 2
                fidelity = ((d1 - 1) * lambda2 + 1 - L1) / d1
                lambda2_err = p0_result.params['lambda2'].stderr or np.nan
                fidelity_err = np.sqrt(lambda2_err ** 2 + L1_err ** 2) / d1
                fit_data['state_0_p0_result'] = p0_result
                fit_data['state_0_p0_fit'] = p0_fit
                fit_data['state_0_lambda2'] = (lambda2, lambda2_err)
                fit_data['state_0_fidelity'] = (fidelity, fidelity_err)
                fit_data['state_0_fidelity_regime'] = fit_regime
            except Exception as e:
                print(f"Fit failed for computational-subspace population: {e}")

            # Keep independent state fits as diagnostics only; they are not used
            # to determine the primary L1/L2 estimates above.
            for i, line in enumerate(fit_lines[:2]):
                try:
                    pop_data = populations[i]
                    if not np.any(pop_data > 0):
                        continue
                    popt, pcov = curve_fit(fit_leakage_RB, self.circuit_depths, pop_data,
                                           p0=[0.01, 0.005, 0.0])
                    line.set_data(self.circuit_depths, fit_leakage_RB(self.circuit_depths, *popt))
                    fit_data[f"state_{i}"] = {"popt": popt, "pcov": pcov}
                except Exception as e:
                    print(f"Diagnostic fit failed for state {i}: {e}")

            # Plot the data live
            ax.relim()
            ax.autoscale_view(scaley=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)
        # Loop is finished, print results
        print("")
        try:
            result = fit_data['state_2_result']
            L1, L1_err = fit_data['state_2_L1']
            L2, L2_err = fit_data['state_2_L2']
            lambda1, lambda1_err = fit_data['state_2_lambda1']
            print("################################################################")
            print(result.fit_report())
            print(f"L1  = {L1:.6g} ± {L1_err:.6g}")
            print(f"L2  = {L2:.6g} ± {L2_err:.6g}")
            print(f"λ1  = {lambda1:.6f} ± {lambda1_err:.6f}")
            lambda2, lambda2_err = fit_data['state_0_lambda2']
            fidelity, fidelity_err = fit_data['state_0_fidelity']
            print(f"λ2  = {lambda2:.6f} ± {lambda2_err:.6f}")
            print(f"F   = {fidelity:.8f} ± {fidelity_err:.8f} ({fit_data['state_0_fidelity_regime']} P0 fit)")
            print("################################################################")
        except Exception:
            pass
        # Save Figure
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

        # Update data dict
        self.data.update({
            "populations": populations,
            "fit_data": {k: v for k, v in fit_data.items()
                         if k not in ('state_2_result', 'state_0_p0_result')},
        })

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
    CIRCUIT_DEPTHS = [8]
    NUM_CIRCUITS = 51
    SHOTS = 1
    SIM_DURATION = 5_000 // 4  # clock cycles (4 ns each)

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = []  # not used during simulation

    exp = RB1Q_Leakage(
        QUBIT, CIRCUIT_DEPTHS, NUM_CIRCUITS, SHOTS, calib_params,
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix='RB1Q_leakage_sim',
        config=config,
        simulate=True,
        sim_duration=SIM_DURATION,
    )