from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.ss_helpers import find_blob_mean, find_threshold
from LabCode2.Helpers.fit_functions import double_gaussian
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg


class TemperatureBoltzmann(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element,shots, calib_params, anharm, outer_folder='',path='SingleShotCal', suffix='data', meta_dict=None, plot=True,save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element name
        self.shots = shots
        self.plot = plot
        self.save = save
        self.calib_params = calib_params
        self.job = self.acquire(self.make_prog())
        self.anharm = anharm    # MHz
        self.analyze()
        if self.save:
            self.pickle_data()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit)
                # align()
                # flat_top_iSWAP('c1')
                align()
                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                save(I, I_stream)
                save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def analyze(self):
        job = self.job
        num_bins = int(np.sqrt(self.shots))

        thresh, c_factor, c_theta = self.calib_params

        I_handle = job.result_handles.get("I")
        Q_handle = job.result_handles.get("Q")

        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()

        I = I_handle.fetch_all()['value']
        Q = Q_handle.fetch_all()['value']
        c = I + 1j * Q
        c_rot = np.exp(-1j * c_theta) * c

        ss_array = super().job_to_shots_1q(self.job, [self.shots], self.calib_params)

        self.data = [range(self.shots), ss_array]
        pop1 = ss_array.mean()
        pop0 = 1-pop1

        kb = 20.83661912    # MHz/mK
        temp = 2*np.pi * self.anharm/(np.log((pop0/pop1)) * kb)

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.hist(np.real(c_rot), bins=num_bins, density=True, histtype='step')

            ax.set_title(f'{self.element} IQ Data, Temperature: {temp} mK')
            ax.set_xlabel('IQ bin (arb.) ')
            ax.set_ylabel('Population (arb.)')
            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)
        return temp

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)