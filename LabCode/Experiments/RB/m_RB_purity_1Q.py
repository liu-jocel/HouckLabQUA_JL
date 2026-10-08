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
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
# Reuse the EXACT proven sequence generator and Clifford player from RB1Q.
from LabCode.Experiments.RB.m_RB_1Q import generate_sequence, play_sequence

u = unit(coerce_to_integer=True)


class RB1Q_Purity(ExperimentClass):
    """
    Simplest purity / unitarity RB.

    This is RB1Q with two changes inside the QUA program:
      1. The recovery (inverse) gate is removed -- we play only the m random
         Cliffords and leave the state where it lands.
      2. Before the Z-basis readout we apply one of three basis-change gates
         (I / Y90 / X90) so the same single-shot readout measures <Z> / <X> / <Y>.
         (Sign is irrelevant; purity squares <P>.)

    The QUA program does nothing more than RB1Q's stream layer already does:
    it averages the single-shot result over shots. The basis is just one extra
    .buffer(3) level, so the saved array is [k, depths, 3].

    ALL purity post-processing happens in numpy in analyze():
        <P> = 1 - 2 * P_excited            (per k, depth, basis)
        purity(k, m) = <Z>^2 + <X>^2 + <Y>^2   (square PER sequence)
        purity(m)    = mean over sequences k
        fit  purity(m) = A * u^(m-1) + B   ->  unitarity u
    """

    def __init__(self, element, circuit_depths, num_circuits, shots, calib_params, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None, simulate=False, sim_duration=10_000, correct_readout_errors=True, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.max_depth = max(circuit_depths)
        self.circuit_depths = circuit_depths
        self.num_circuits = num_circuits
        self.shots = shots
        self.calib_params = calib_params
        self.seed = rd.randint(1, 1000000)
        self.correct_readout_errors = correct_readout_errors

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
            m = declare(int)
            mm1 = declare(int)
            b = declare(int)
            n = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            k_stream = declare_stream()

            with for_(k, 0, k < self.num_circuits, k + 1):
                sequence_list, _ = generate_sequence(self.seed, self.max_depth)

                with for_each_(m, self.circuit_depths):
                    # play_sequence plays indices 0..depth, so to play m random
                    # Cliffords with NO recovery gate, pass depth = m - 1.
                    assign(mm1, m - 1)

                    with for_each_(b, [0, 1, 2]):
                        with for_(n, 0, n < self.shots, n + 1):
                            wait(resonator_dump_time * u.ns, self.element)
                            reset_frame(qubit)
                            align()

                            # m random Cliffords, no inverse
                            play_sequence(qubit, sequence_list, mm1)

                            # basis-change gate (identity is wait(4), matching the
                            # RB1Q identity-Clifford convention)
                            with switch_(b, unsafe=True):
                                with case_(0):
                                    wait(4, qubit)       # I   -> measures <Z>
                                with case_(1):
                                    play("Y90", qubit)   # Y90 -> measures <X>
                                with case_(2):
                                    play("X90", qubit)   # X90 -> measures <Y>

                            # SS readout (identical to RB1Q)
                            align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            with if_(I > ss_thresh):
                                assign(state, 1)
                            with else_():
                                assign(state, 0)
                            save(state, ss_stream)
                            align(qubit, resonator)

                            # Active reset (identical to RB1Q)
                            with while_(I_reset > ground_confidence_threshold):
                                with if_(I_reset > ss_thresh):
                                    play('X180', qubit)
                                    align(qubit, resonator)
                                I, Q = measure_1q(self.meta_dict, I, Q)
                                assign(I_reset, I)
                save(k, k_stream)

            with stream_processing():
                k_stream.save("iteration")
                # Same pattern as RB1Q, with one extra buffer(3) for the basis.
                # Result shape: [k, n_depths, 3]  (last axis = basis 0:Z, 1:X, 2:Y)
                ss_stream.buffer(self.shots).map(FUNCTIONS.average()).buffer(3).buffer(
                    len(self.circuit_depths)).save_all("ss")
        return prog

    def make_sim_prog(self):
        """Single random purity sequence (longest depth) for waveform simulation."""
        qubit = self.meta_dict['q_name']
        depth = self.circuit_depths[-1]
        with program() as prog:
            seq_stream = declare_stream()
            i = declare(int)
            mm1 = declare(int)
            sequence_list, _ = generate_sequence(self.seed, self.max_depth)
            with for_(i, 0, i < depth, i + 1):
                save(sequence_list[i], seq_stream)
            reset_frame(qubit)
            assign(mm1, depth - 1)
            play_sequence(qubit, sequence_list, mm1)
            play("Y90", qubit)
            with stream_processing():
                seq_stream.save_all("sim_sequence")
        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss", "iteration"], mode="live")

        def fitU(x, A, uu, B):
            return A * uu ** (x - 1) + B

        depths = np.array(self.circuit_depths, dtype=float)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        line, = ax.plot([], [], 'o', label='purity')
        fit_line, = ax.plot([], [], '--', color='C0', alpha=0.7, label='fit')
        ax.set_xlabel('Num Cliffords')
        ax.set_ylabel('Purity  <X>^2 + <Y>^2 + <Z>^2')
        ax.set_xscale('log')
        ax.set_xlim(self.circuit_depths[0], self.circuit_depths[-1])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which='both')
        ax.legend()
        fig.suptitle(f"Purity RB on {self.element}")
        interrupt_on_close(fig, job)

        purity_m = None
        while results.is_processing():
            ss_all, iteration = results.fetch_all()
            if ss_all.dtype.names:
                ss_all = ss_all['value']
            if ss_all.shape[0] == 0:
                continue
            print(f"Circuit: {iteration}/{self.num_circuits}", end="\r", flush=True)
            # ss_all: [k_elapsed, n_depths, 3] = excited fraction per (seq, depth, basis)
            # ---- all purity post-processing in numpy ----
            P = 1.0 - 2.0 * ss_all                 # <P> in each basis
            purity_km = np.sum(P ** 2, axis=-1)    # square PER sequence, sum over basis
            purity_m = np.mean(purity_km, axis=0)  # THEN average over sequences
            line.set_data(depths, purity_m)
            try:
                popt, _ = curve_fit(fitU, depths, purity_m, p0=[0.8, 0.99, 0.0],
                                    bounds=([0.0, 0.0, -0.5], [3.0, 1.0, 0.5]), maxfev=10000)
                fit_x = np.linspace(depths[0], depths[-1], 200)
                fit_line.set_data(fit_x, fitU(fit_x, *popt))
                ax.set_title(f"u = {popt[1]:.6f}")
            except Exception:
                pass
            ax.relim()
            ax.autoscale_view(scaley=True)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        # Final fit
        if purity_m is not None:
            try:
                popt, pcov = curve_fit(fitU, depths, purity_m, p0=[0.8, 0.99, 0.0],
                                       bounds=([0.0, 0.0, -0.5], [3.0, 1.0, 0.5]), maxfev=10000)
                u_fit = float(popt[1])
                u_err = float(np.sqrt(pcov[1][1]))
                print(f"Purity RB unitarity: u = {u_fit:.6f} +/- {u_err:.6f}")
                self.data.update({
                    'unitarity': u_fit,
                    'unitarity_err': u_err,
                    'purity_fit_popt': popt,
                    'purity_fit_pcov': pcov,
                })
            except Exception:
                print("Purity RB fit failed.")
                self.data.update({'unitarity': np.nan, 'unitarity_err': np.nan})

        self.data.update({'purity_decay': purity_m})

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()
        return purity_m


if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    QUBIT = 'q4'
    CIRCUIT_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    NUM_CIRCUITS = 100
    SHOTS = 100

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {"threshold": 0, "ground_threshold": 0}

    exp = RB1Q_Purity(
        QUBIT, CIRCUIT_DEPTHS, NUM_CIRCUITS, SHOTS, calib_params,
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix='PurityRB',
        config=config,
        simulate=False,
    )