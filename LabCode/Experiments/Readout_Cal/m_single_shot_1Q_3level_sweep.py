from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q_amp
from LabCode.Helpers.ss_helpers import find_blob_median, find_threshold, find_state
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class SingleShot1Q3LevelSweep(ExperimentClass):
    """
    3-level single-shot readout calibration with in-QUA sweep over resonator
    frequency (update_frequency) and readout amplitude (amp() scaling).

    All (a, df) combinations are swept inside a single QUA program — no new
    config or meta dict is needed per point. States |0>, |1>, |2> are prepared
    and measured at each (a, df) point within the same shot loop, enabling
    live fidelity heatmap updates as shots accumulate.

    Parameters
    ----------
    a_vec : array of absolute readout amplitudes (V). Normalized internally
            by meta_dict['read_amp'].
    f_vec : array of resonator IF frequency detunings (Hz) around meta_dict['r_IF'].
    """

    def __init__(self, element, a_vec, f_vec, shots,
                 outer_folder='', path='SingleShotCal', suffix='data',
                 meta_dict=None, config=None, save_data=True, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config, **kwargs)

        self.element = element
        self.a_vec = np.array(a_vec)
        self.f_vec = np.array(f_vec)
        self.shots = shots
        self.a_vec_norm = self.a_vec / meta_dict['read_amp']

        self.data = {
            'qubit': self.element,
            'a_vec': self.a_vec,
            'f_vec': self.f_vec,
            'shots': self.shots,
            'meta_dict': self.meta_dict,
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.max_F = self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if save_data:
            self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        r_IF = self.meta_dict['r_IF']
        IF_freq = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        anharmonicity = self.meta_dict['anharmonicity']

        with program() as prog:
            n = declare(int)
            a = declare(fixed)
            df = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()
            I_stream_2 = declare_stream()
            Q_stream_2 = declare_stream()

            with for_(n, 0, n < self.shots, n + 1):
                with for_(*from_array(a, self.a_vec_norm)):
                    with for_each_(df, self.f_vec):
                        update_frequency(resonator, r_IF + df)

                        # |0>
                        wait(reset_time * u.ns, qubit)
                        align()
                        I, Q = measure_1q_amp(self.meta_dict, a, I, Q)
                        save(I, I_stream_0)
                        save(Q, Q_stream_0)

                        # |1>
                        wait(reset_time * u.ns, qubit)
                        align()
                        play('X180', qubit)
                        align()
                        I, Q = measure_1q_amp(self.meta_dict, a, I, Q)
                        save(I, I_stream_1)
                        save(Q, Q_stream_1)

                        # |2>
                        wait(reset_time * u.ns, qubit)
                        align()
                        play('X180', qubit)
                        align()
                        update_frequency(qubit, IF_freq + anharmonicity)
                        play('X180_ef', qubit)
                        update_frequency(qubit, IF_freq)
                        align()
                        I, Q = measure_1q_amp(self.meta_dict, a, I, Q)
                        save(I, I_stream_2)
                        save(Q, Q_stream_2)

            with stream_processing():
                I_stream_0.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("I_0")
                Q_stream_0.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("Q_0")
                I_stream_1.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("I_1")
                Q_stream_1.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("Q_1")
                I_stream_2.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("I_2")
                Q_stream_2.buffer(len(self.f_vec)).buffer(len(self.a_vec)).save_all("Q_2")

        return prog

    def analyze(self, job, qm):
        n_a = len(self.a_vec)
        n_f = len(self.f_vec)

        results = fetching_tool(
            job, data_list=["I_0", "Q_0", "I_1", "Q_1", "I_2", "Q_2"], mode="live"
        )

        fidelities = np.full((n_a, n_f), 0.5)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        pcm = ax.pcolormesh(
            self.f_vec / u.MHz, self.a_vec, fidelities,
            vmin=0.5, vmax=1.0, shading='nearest'
        )
        fig.colorbar(pcm, ax=ax, label='Fidelity  0.5·(F₀₁ + F₁₂)')
        ax.set_xlabel('Resonator detuning (MHz)')
        ax.set_ylabel('Readout amplitude (V)')
        ax.set_title(f'{self.element}  3-level readout calibration sweep')
        star = ax.scatter([], [], marker='*', s=300, c='red', zorder=5)
        interrupt_on_close(fig, job)

        I_0_all = Q_0_all = I_1_all = Q_1_all = I_2_all = Q_2_all = None

        while results.is_processing():
            I_0_all, Q_0_all, I_1_all, Q_1_all, I_2_all, Q_2_all = results.fetch_all()
            n = min(
                I_0_all.shape[0], Q_0_all.shape[0],
                I_1_all.shape[0], Q_1_all.shape[0],
                I_2_all.shape[0], Q_2_all.shape[0],
            )
            if n == 0:
                continue
            I_0_s = I_0_all[:n]
            Q_0_s = Q_0_all[:n]
            I_1_s = I_1_all[:n]
            Q_1_s = Q_1_all[:n]
            I_2_s = I_2_all[:n]
            Q_2_s = Q_2_all[:n]

            for i in range(n_a):
                for j in range(n_f):
                    c_0 = I_0_s[:, i, j] + 1j * Q_0_s[:, i, j]
                    c_1 = I_1_s[:, i, j] + 1j * Q_1_s[:, i, j]
                    c_2 = I_2_s[:, i, j] + 1j * Q_2_s[:, i, j]
                    try:
                        th01 = np.angle(find_blob_median(c_1) - find_blob_median(c_0))
                        th12 = np.angle(find_blob_median(c_2) - find_blob_median(c_1))
                        _, F_01 = find_threshold(
                            np.exp(-1j * th01) * c_0, np.exp(-1j * th01) * c_1
                        )
                        _, F_12 = find_threshold(
                            np.exp(-1j * th12) * c_1, np.exp(-1j * th12) * c_2
                        )
                        fidelities[i, j] = 0.5 * (np.max(F_01) + np.max(F_12))
                    except Exception:
                        pass

            best_idx = np.unravel_index(np.argmax(fidelities), fidelities.shape)
            best_f_MHz = self.f_vec[best_idx[1]] / u.MHz
            best_a = self.a_vec[best_idx[0]]
            pcm.set_array(fidelities.ravel())
            pcm.set_clim(max(fidelities.min(), 0.5), fidelities.max())
            star.set_offsets([[best_f_MHz, best_a]])
            ax.set_title(
                f'{self.element}  best F={fidelities[best_idx]:.3f}  '
                f'a={best_a:.4f} V  df={best_f_MHz:.2f} MHz  '
                f'({I_0_all.shape[0]}/{self.shots} shots)'
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        # Guard against early interrupt with no data
        if I_0_all is None or I_0_all.shape[0] == 0:
            qm.close()
            plt.close(fig)
            plt.ioff()
            return 0.0

        # --- Full confusion matrix at the optimal (a, df) point ---
        best_idx = np.unravel_index(np.argmax(fidelities), fidelities.shape)
        ai, fi = best_idx
        optimal_amp = self.a_vec[ai]
        optimal_df = int(self.f_vec[fi])

        I_0_opt = I_0_all[:, ai, fi];  Q_0_opt = Q_0_all[:, ai, fi]
        I_1_opt = I_1_all[:, ai, fi];  Q_1_opt = Q_1_all[:, ai, fi]
        I_2_opt = I_2_all[:, ai, fi];  Q_2_opt = Q_2_all[:, ai, fi]

        IQ_0 = [np.mean(I_0_opt), np.mean(Q_0_opt)]
        IQ_1 = [np.mean(I_1_opt), np.mean(Q_1_opt)]
        IQ_2 = [np.mean(I_2_opt), np.mean(Q_2_opt)]

        n_shots = I_0_all.shape[0]
        confusion = np.zeros([3, 3])
        for k in range(n_shots):
            confusion[find_state([I_0_opt[k], Q_0_opt[k]], IQ_0, IQ_1, IQ_2)][0] += 1 / n_shots
            confusion[find_state([I_1_opt[k], Q_1_opt[k]], IQ_0, IQ_1, IQ_2)][1] += 1 / n_shots
            confusion[find_state([I_2_opt[k], Q_2_opt[k]], IQ_0, IQ_1, IQ_2)][2] += 1 / n_shots

        max_F = np.trace(confusion) / 3
        print(f'Optimal: amp={optimal_amp:.4f} V,  df={optimal_df / u.MHz:.3f} MHz')
        print(f'Confusion matrix:\n{confusion}')
        print(f'Average fidelity: {max_F:.4f}')

        self.data.update({
            'fidelities': fidelities,
            'optimal_amp': optimal_amp,
            'optimal_f_detuning': optimal_df,
            'confusion': confusion,
            'cs': np.array([
                I_0_opt + 1j * Q_0_opt,
                I_1_opt + 1j * Q_1_opt,
                I_2_opt + 1j * Q_2_opt,
            ]),
        })

        qm.close()
        fig.savefig(self.iname, bbox_inches='tight')
        plt.close(fig)
        plt.ioff()
        return max_F

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])
