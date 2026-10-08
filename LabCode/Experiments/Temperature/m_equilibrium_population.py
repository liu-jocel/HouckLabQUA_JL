from qm.qua import *
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import find_blob_mean, find_threshold, find_blob_median
from qualang_tools.plot import interrupt_on_close
from scipy.stats import norm
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
class EquilibriumPopulation(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self, element, shots, outer_folder='', path='SingleShotCal', suffix='data', meta_dict=None, config=None, save=True,
                 plot=True, repeats=1, confidence_threshold=6e-1):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.repeats = repeats
        self.confidence_threshold = confidence_threshold
        self.shots = shots
        self.meta_dict = meta_dict
        self.config=config
        self.plot = plot
        self.save = save
        self.data = {
            'meta_dict': meta_dict,
            'shots': shots,
            'repeats': self.repeats,
        }

        self.job, self.qm = self.acquire(self.make_prog(), config=config, return_qm=True)
        self.max_F = self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if self.save:
            self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_LO']["LO_name"]
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()


            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit)
                align(qubit, resonator)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_0)
                save(Q, Q_stream_0)
            with stream_processing():
                I_stream_0.save_all("I_0")
                Q_stream_0.save_all("Q_0")


        return prog

    def analyze(self):
        job = self.job

        h_I = job.result_handles.get("I_0")
        h_Q = job.result_handles.get("Q_0")

        if self.plot:
            matplotlib.use("qtagg", force=True)
            plt.ion()
            fig_live, ax_live = plt.subplots(1, 1, constrained_layout=True)
            sc = ax_live.scatter([], [], s=1, alpha=0.5, color='C0')
            ax_live.set_xlabel('I')
            ax_live.set_ylabel('Q')
            ax_live.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — acquiring...')
            interrupt_on_close(fig_live, job)

        while h_I.count_so_far() < self.shots:
            if self.plot:
                n = h_I.count_so_far()
                if n > 0:
                    I_live = h_I.fetch_all()['value']
                    Q_live = h_Q.fetch_all()['value']
                    n = min(len(I_live), len(Q_live))
                    sc.set_offsets(np.c_[I_live[:n], Q_live[:n]])
                    all_pts = sc.get_offsets()
                    pad = max(all_pts[:, 0].ptp() or 1, all_pts[:, 1].ptp() or 1) * 0.1
                    ax_live.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
                    ax_live.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
                ax_live.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — {h_I.count_so_far()}/{self.shots} shots')
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(0.05)

        h_I.wait_for_all_values()
        h_Q.wait_for_all_values()

        I_0 = h_I.fetch_all()['value']
        Q_0 = h_Q.fetch_all()['value']

        if self.plot:
            sc.set_offsets(np.c_[I_0, Q_0])
            ax_live.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — done ({self.shots} shots)')
            fig_live.canvas.draw_idle()
            fig_live.canvas.flush_events()
            plt.pause(0.1)
            plt.ioff()

        self.data.update({"I": I_0, "Q": Q_0})

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.scatter(I_0, Q_0, s=1, color='black', alpha=0.5)
            ax.set_title(f'{self.element}  ro_amp={self.meta_dict["read_amp"]:.4f} V — equilibrium population')
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            plt.savefig(self.iname[:-4] + "_blob", bbox_inches='tight')
            plt.close(fig)

        self.qm.close()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])