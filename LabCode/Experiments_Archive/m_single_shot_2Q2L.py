from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode2.Helpers.ss_helpers import find_blob_mean, find_threshold, concatenate, find_state
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg

class SingleShot2Q2L(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element_1,element_2,shots,outer_folder='',path='SingleShotCal', suffix='2Q2L', meta_dict=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element_1 = element_1
        self.element_2 = element_2
        self.shots = shots
        self.plot = plot

        self.job = self.acquire(self.make_prog())
        self.max_F = self.analyze()

    def make_prog(self):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_stream_c1_00 = declare_stream()
            Q_stream_c1_00 = declare_stream()
            I_stream_c2_00 = declare_stream()
            Q_stream_c2_00 = declare_stream()
            I_stream_c1_01 = declare_stream()
            Q_stream_c1_01 = declare_stream()
            I_stream_c2_01 = declare_stream()
            Q_stream_c2_01 = declare_stream()
            I_stream_c1_10 = declare_stream()
            Q_stream_c1_10 = declare_stream()
            I_stream_c2_10 = declare_stream()
            Q_stream_c2_10 = declare_stream()
            I_stream_c1_11 = declare_stream()
            Q_stream_c1_11 = declare_stream()
            I_stream_c2_11 = declare_stream()
            Q_stream_c2_11 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                # align()
                # flat_top_iSWAP('c1')
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_00)
                save(Q1, Q_stream_c1_00)
                save(I2, I_stream_c2_00)
                save(Q2, Q_stream_c2_00)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                # align()
                # flat_top_iSWAP('c1')
                align()
                play('X180', qubit_2)
                align()
                # play('iSWAP_const' * amp(1), 'c1')
                # align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_01)
                save(Q1, Q_stream_c1_01)
                save(I2, I_stream_c2_01)
                save(Q2, Q_stream_c2_01)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                # align()
                # flat_top_iSWAP('c1')
                align()
                play('X180', qubit_1)
                # align()
                # play('iSWAP_const' * amp(1), 'c1')
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_10)
                save(Q1, Q_stream_c1_10)
                save(I2, I_stream_c2_10)
                save(Q2, Q_stream_c2_10)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                # align()
                # flat_top_iSWAP('c1')
                align()
                play('X180', qubit_1)
                play('X180', qubit_2)
                # align()
                # play('iSWAP_const' * amp(1), 'c1')
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_11)
                save(Q1, Q_stream_c1_11)
                save(I2, I_stream_c2_11)
                save(Q2, Q_stream_c2_11)


            with stream_processing():

                I_stream_c1_00.save_all('I_c1_00')
                Q_stream_c1_00.save_all('Q_c1_00')
                I_stream_c2_00.save_all('I_c2_00')
                Q_stream_c2_00.save_all('Q_c2_00')
                I_stream_c1_01.save_all('I_c1_01')
                Q_stream_c1_01.save_all('Q_c1_01')
                I_stream_c2_01.save_all('I_c2_01')
                Q_stream_c2_01.save_all('Q_c2_01')
                I_stream_c1_10.save_all('I_c1_10')
                Q_stream_c1_10.save_all('Q_c1_10')
                I_stream_c2_10.save_all('I_c2_10')
                Q_stream_c2_10.save_all('Q_c2_10')
                I_stream_c1_11.save_all('I_c1_11')
                Q_stream_c1_11.save_all('Q_c1_11')
                I_stream_c2_11.save_all('I_c2_11')
                Q_stream_c2_11.save_all('Q_c2_11')

        return prog

    def analyze(self):

        job = self.job

        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        I_c1_00 = res_handles.get("I_c1_00").fetch_all()['value']
        Q_c1_00 = res_handles.get("Q_c1_00").fetch_all()['value']
        I_c2_00 = res_handles.get("I_c2_00").fetch_all()['value']
        Q_c2_00 = res_handles.get("Q_c2_00").fetch_all()['value']

        I_c1_01 = res_handles.get("I_c1_01").fetch_all()['value']
        Q_c1_01 = res_handles.get("Q_c1_01").fetch_all()['value']
        I_c2_01 = res_handles.get("I_c2_01").fetch_all()['value']
        Q_c2_01 = res_handles.get("Q_c2_01").fetch_all()['value']

        I_c1_10 = res_handles.get("I_c1_10").fetch_all()['value']
        Q_c1_10 = res_handles.get("Q_c1_10").fetch_all()['value']
        I_c2_10 = res_handles.get("I_c2_10").fetch_all()['value']
        Q_c2_10 = res_handles.get("Q_c2_10").fetch_all()['value']

        I_c1_11 = res_handles.get("I_c1_11").fetch_all()['value']
        Q_c1_11 = res_handles.get("Q_c1_11").fetch_all()['value']
        I_c2_11 = res_handles.get("I_c2_11").fetch_all()['value']
        Q_c2_11 = res_handles.get("Q_c2_11").fetch_all()['value']

        IQ_c1_00 = [np.mean(I_c1_00), np.mean(Q_c1_00)]
        IQ_c2_00 = [np.mean(I_c2_00), np.mean(Q_c2_00)]
        IQ_c1_01 = [np.mean(I_c1_01), np.mean(Q_c1_01)]
        IQ_c2_01 = [np.mean(I_c2_01), np.mean(Q_c2_01)]
        IQ_c1_10 = [np.mean(I_c1_10), np.mean(Q_c1_10)]
        IQ_c2_10 = [np.mean(I_c2_10), np.mean(Q_c2_10)]
        IQ_c1_11 = [np.mean(I_c1_11), np.mean(Q_c1_11)]
        IQ_c2_11 = [np.mean(I_c2_11), np.mean(Q_c2_11)]

        num_shots = self.shots
        self.confusion = np.zeros([4, 4])

        for i in range(num_shots):
            s_c1_00 = find_state([I_c1_00[i], Q_c1_00[i]], IQ_c1_00 + IQ_c1_01, IQ_c1_10 + IQ_c1_11)
            s_c2_00 = find_state([I_c2_00[i], Q_c2_00[i]], IQ_c2_00 + IQ_c2_10, IQ_c2_01 + IQ_c2_11)
            self.confusion[2 * s_c1_00 + s_c2_00][0] += 1 / num_shots

            s_c1_01 = find_state([I_c1_01[i], Q_c1_01[i]], IQ_c1_00 + IQ_c1_01, IQ_c1_10 + IQ_c1_11)
            s_c2_01 = find_state([I_c2_01[i], Q_c2_01[i]], IQ_c2_00 + IQ_c2_10, IQ_c2_01 + IQ_c2_11)
            self.confusion[2 * s_c1_01 + s_c2_01][1] += 1 / num_shots

            s_c1_10 = find_state([I_c1_10[i], Q_c1_10[i]], IQ_c1_00 + IQ_c1_01, IQ_c1_10 + IQ_c1_11)
            s_c2_10 = find_state([I_c2_10[i], Q_c2_10[i]], IQ_c2_00 + IQ_c2_10, IQ_c2_01 + IQ_c2_11)
            self.confusion[2 * s_c1_10 + s_c2_10][2] += 1 / num_shots

            s_c1_11 = find_state([I_c1_11[i], Q_c1_11[i]], IQ_c1_00 + IQ_c1_01, IQ_c1_10 + IQ_c1_11)
            s_c2_11 = find_state([I_c2_11[i], Q_c2_11[i]], IQ_c2_00 + IQ_c2_10, IQ_c2_01 + IQ_c2_11)
            self.confusion[2 * s_c1_11 + s_c2_11][3] += 1 / num_shots

        self.calib_params = [IQ_c1_00, IQ_c2_00, IQ_c1_01, IQ_c2_01, IQ_c1_10, IQ_c2_10, IQ_c1_11, IQ_c2_11]

        if self.plot:
            fig, ax = plt.subplots(2, 1, constrained_layout=True)

            ax[0].plot(Q_c1_00, I_c1_00, '.', label='00')
            ax[0].plot(Q_c1_10, I_c1_10, '.', label='10')
            ax[0].plot(Q_c1_01, I_c1_01, '.', label='01')
            ax[0].plot(Q_c1_11, I_c1_11, '.', label='11')

            ax[1].plot(Q_c2_00, I_c2_00, '.', label='00')
            ax[1].plot(Q_c2_10, I_c2_10, '.', label='10')
            ax[1].plot(Q_c2_01, I_c2_01, '.', label='01')
            ax[1].plot(Q_c2_11, I_c2_11, '.', label='11')

            ax[0].legend()
            ax[1].legend()

            ax[0].set_title(f'{self.element_1}')
            ax[1].set_title(f'{self.element_2}')

            ax[0].set_ylabel('I')
            ax[0].set_xlabel('Q')
            ax[1].set_ylabel('I')
            ax[1].set_xlabel('Q')

            fig.suptitle(f'F = {np.trace(self.confusion) / 4:.3f}')

            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

        return np.trace(self.confusion) / 2
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])