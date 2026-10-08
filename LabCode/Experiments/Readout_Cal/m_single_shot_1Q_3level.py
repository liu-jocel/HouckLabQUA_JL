from qm.qua import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode.Helpers.ss_helpers import find_blob_mean, find_threshold, find_blob_median, find_state
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
class SingleShot1Q3Level(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element,shots,outer_folder='',path='SingleShotCal', suffix='data', meta_dict=None, config=None, plot=True, save_data=True, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)
        self.element = element #meta dict element name
        self.shots = shots
        self.data = {"shots": self.shots,
                     "meta_dict": self.meta_dict,}
        self.plot = plot

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.max_F = self.analyze()
        if save_data:
            self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        IF_freq = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        anharmonicity = self.meta_dict['anharmonicity']

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()
            I_stream_2 = declare_stream()
            Q_stream_2 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                # |0⟩
                wait(reset_time*u.ns, qubit)
                align()
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_0)
                save(Q, Q_stream_0)

                # |1⟩
                wait(reset_time*u.ns, qubit)
                align()
                play('X180', qubit)
                align()
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_1)
                save(Q, Q_stream_1)

                # |2⟩
                wait(reset_time*u.ns, qubit)
                align()
                play('X180', qubit)
                align()
                update_frequency(qubit, IF_freq + anharmonicity)
                play('X180_ef', qubit)
                update_frequency(qubit, IF_freq)
                align()
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_2)
                save(Q, Q_stream_2)

            with stream_processing():
                I_stream_0.save_all("I_0")
                Q_stream_0.save_all("Q_0")
                I_stream_1.save_all("I_1")
                Q_stream_1.save_all("Q_1")
                I_stream_2.save_all("I_2")
                Q_stream_2.save_all("Q_2")

        return prog

    def analyze(self):
        '''
        Function to analyze raw SSCAL data
        '''
        job = self.job

        h_I0 = job.result_handles.get("I_0")
        h_Q0 = job.result_handles.get("Q_0")
        h_I1 = job.result_handles.get("I_1")
        h_Q1 = job.result_handles.get("Q_1")
        h_I2 = job.result_handles.get("I_2")
        h_Q2 = job.result_handles.get("Q_2")

        # Live scatter
        if self.plot:
            matplotlib.use("qtagg", force=True)
            plt.ion()
            fig_live, ax_live = plt.subplots(1, 1, constrained_layout=True)
            sc0 = ax_live.scatter([], [], s=1, alpha=0.5, label='|0⟩', color='C0')
            sc1 = ax_live.scatter([], [], s=1, alpha=0.5, label='|1⟩', color='C1')
            sc2 = ax_live.scatter([], [], s=1, alpha=0.5, label='|2⟩', color='C2')
            ax_live.set_xlabel('I')
            ax_live.set_ylabel('Q')
            ax_live.legend()
            interrupt_on_close(fig_live, job)

        stream_pairs = [(h_I0, h_Q0, sc0), (h_I1, h_Q1, sc1), (h_I2, h_Q2, sc2)] if self.plot else []

        def n_done():
            return h_I0.count_so_far() + h_I1.count_so_far() + h_I2.count_so_far()

        while n_done() < 3 * self.shots:
            if self.plot:
                for h_I, h_Q, sc in stream_pairs:
                    if h_I.count_so_far() > 0:
                        I_live = h_I.fetch_all()['value']
                        Q_live = h_Q.fetch_all()['value']
                        n = min(len(I_live), len(Q_live))
                        sc.set_offsets(np.c_[I_live[:n], Q_live[:n]])
                ax_live.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — {n_done()}/{3 * self.shots} shots')
                nonempty = [sc.get_offsets() for sc in (sc0, sc1, sc2)
                            if len(sc.get_offsets()) > 0]
                if nonempty:
                    all_pts = np.vstack(nonempty)
                    pad = max((all_pts[:, 0].ptp() or 1), (all_pts[:, 1].ptp() or 1)) * 0.1
                    ax_live.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
                    ax_live.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(0.05)

        for h in (h_I0, h_Q0, h_I1, h_Q1, h_I2, h_Q2):
            h.wait_for_all_values()

        I_0 = h_I0.fetch_all()['value']
        Q_0 = h_Q0.fetch_all()['value']
        I_1 = h_I1.fetch_all()['value']
        Q_1 = h_Q1.fetch_all()['value']
        I_2 = h_I2.fetch_all()['value']
        Q_2 = h_Q2.fetch_all()['value']

        if self.plot:
            for h_I, h_Q, sc in stream_pairs:
                sc.set_offsets(np.c_[h_I.fetch_all()['value'], h_Q.fetch_all()['value']])
            ax_live.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — done ({3 * self.shots} shots)')
            fig_live.canvas.draw_idle()
            fig_live.canvas.flush_events()
            plt.pause(0.1)
            plt.ioff()
            plt.close(fig_live)

        # combine IQ data
        c_0 = I_0 + 1j * Q_0
        c_1 = I_1 + 1j * Q_1
        c_2 = I_2 + 1j * Q_2

        self.data.update({'cs': np.array([c_0, c_1, c_2])})

        # Calculate Rotation Angles
        c_theta_01 = np.angle(find_blob_median(c_1) - find_blob_median(c_0))
        c_theta_21 = np.angle(find_blob_median(c_2) - find_blob_median(c_1))
        c_theta_20 = np.angle(find_blob_median(c_2) - find_blob_median(c_0))

        # Rotate Data
        c_0_rot_01_discrim = np.exp(-1j * c_theta_01) * c_0     # |0> rotated for discrim along |0> - |1> line
        c_1_rot_01_discrim = np.exp(-1j * c_theta_01) * c_1     # |1> rotated for discrim along |0> - |1> line

        c_1_rot_21_discrim = np.exp(-1j * c_theta_21) * c_1
        c_2_rot_21_discrim = np.exp(-1j * c_theta_21) * c_2

        c_0_rot_02_discrim = np.exp(-1j * c_theta_20) * c_0
        c_2_rot_02_discrim = np.exp(-1j * c_theta_20) * c_2


        # Calculate thresholds for discrimination
        # Test: is data the hypothesis state or not
        c_thresh_01, c_F_01 = find_threshold(c_1_rot_01_discrim, c_0_rot_01_discrim)
        c_thresh_12, c_F_12 = find_threshold(c_2_rot_21_discrim, c_1_rot_21_discrim)
        c_thresh_02, c_F_02 = find_threshold(c_2_rot_02_discrim, c_0_rot_02_discrim)

        thresh_01 = c_thresh_01[np.argmax(c_F_01)]
        thresh_12 = c_thresh_12[np.argmax(c_F_12)]
        thresh_02 = c_thresh_02[np.argmax(c_F_02)]

        self.data.update({
            "01_discrim": {
                "Threshold": thresh_01,
                "Fidelity": c_F_01,
                "Theta": c_theta_01,
            },
            "12_discrim": {
                "Threshold": thresh_12,
                "Fidelity": c_F_12,
                "Theta": c_theta_21,
            },
            "02_discrim": {
                "Threshold": thresh_02,
                "Fidelity": c_F_02,
                "Theta": c_theta_20,
            },
        })

        IQ_0 = [np.mean(I_0), np.mean(Q_0)]
        IQ_1 = [np.mean(I_1), np.mean(Q_1)]
        IQ_2 = [np.mean(I_2), np.mean(Q_2)]

        num_shots = self.shots

        self.confusion = np.zeros([3, 3])

        for i in range(num_shots):

            self.confusion[find_state([I_0[i], Q_0[i]], IQ_0, IQ_1, IQ_2)][0] += 1 / num_shots
            self.confusion[find_state([I_1[i], Q_1[i]], IQ_0, IQ_1, IQ_2)][1] += 1 / num_shots
            self.confusion[find_state([I_2[i], Q_2[i]], IQ_0, IQ_1, IQ_2)][2] += 1 / num_shots

        self.calib_params = [IQ_0, IQ_1, IQ_2]

        print(self.confusion)
        print(np.trace(self.confusion)/3)


        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)

            ax.plot(I_0, Q_0, '.', label='0', alpha=0.5)
            ax.plot(I_1, Q_1, '.', label='1', alpha=0.5)
            ax.plot(I_2, Q_2, '.', label='2', alpha=0.5)

            ax.legend()

            ax.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V  F={np.trace(self.confusion) / 3:.3f}')

            ax.set_ylabel('I')
            ax.set_xlabel('Q')

            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

            self.data.update({
                "I_0": I_0,
                "Q_0": Q_0,
                "I_1": I_1,
                "Q_1": Q_1,
                "I_2": I_2,
                "Q_2": Q_2,
                "Confusion_Matrix": self.confusion
            })

        self.qm.close()
        return np.trace(self.confusion)/3

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])