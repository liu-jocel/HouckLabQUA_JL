from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.ss_helpers import find_blob_mean, find_threshold, find_state
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg

class SingleShot1Q3Level(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element,shots,outer_folder='',path='SingleShotCal', suffix='data', meta_dict=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element name
        self.shots = shots
        self.plot = plot

        self.job = self.acquire(self.make_prog())
        self.max_F = self.analyze()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        shots = self.shots

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
                wait(reset_ticks, qubit)
                align()
                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                save(I, I_stream_0)
                save(Q, Q_stream_0)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit)
                align()
                play('X180', qubit)
                align()
                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                save(I, I_stream_1)
                save(Q, Q_stream_1)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit)
                align()
                play('X180', qubit)
                align()
                update_frequency(qubit, self.meta_dict[qubit]['f_state_IF'])
                play('X180_2', qubit)
                update_frequency(qubit, self.meta_dict[qubit]['q_IF'])
                align()
                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

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

        job = self.job

        I_handle_0 = job.result_handles.get("I_0")
        Q_handle_0 = job.result_handles.get("Q_0")
        I_handle_1 = job.result_handles.get("I_1")
        Q_handle_1 = job.result_handles.get("Q_1")
        I_handle_2 = job.result_handles.get("I_2")
        Q_handle_2 = job.result_handles.get("Q_2")

        I_handle_0.wait_for_all_values()
        Q_handle_0.wait_for_all_values()
        I_handle_1.wait_for_all_values()
        Q_handle_1.wait_for_all_values()

        I_0 = I_handle_0.fetch_all()['value']
        Q_0 = Q_handle_0.fetch_all()['value']
        I_1 = I_handle_1.fetch_all()['value']
        Q_1 = Q_handle_1.fetch_all()['value']
        I_2 = I_handle_2.fetch_all()['value']
        Q_2 = Q_handle_2.fetch_all()['value']

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

            ax.plot(I_0, Q_0, '.', label='0')
            ax.plot(I_1, Q_1, '.', label='1')
            ax.plot(I_2, Q_2, '.', label='2')

            ax.legend()

            ax.set_title(f'{self.element} F = {np.trace(self.confusion) / 3:.3f}')

            ax.set_ylabel('I')
            ax.set_xlabel('Q')

            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

        return np.trace(self.confusion)/3

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])