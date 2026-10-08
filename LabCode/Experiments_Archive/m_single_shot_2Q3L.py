from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode2.Helpers.ss_helpers import find_blob_mean, find_threshold, concatenate, find_state
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg

class SingleShot2Q3L(ExperimentClass):
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
            I_stream_c1_20 = declare_stream()
            Q_stream_c1_20 = declare_stream()
            I_stream_c2_20 = declare_stream()
            Q_stream_c2_20 = declare_stream()
            I_stream_c1_02 = declare_stream()
            Q_stream_c1_02 = declare_stream()
            I_stream_c2_02 = declare_stream()
            Q_stream_c2_02 = declare_stream()
            I_stream_c1_21 = declare_stream()
            Q_stream_c1_21 = declare_stream()
            I_stream_c2_21 = declare_stream()
            Q_stream_c2_21 = declare_stream()
            I_stream_c1_12 = declare_stream()
            Q_stream_c1_12 = declare_stream()
            I_stream_c2_12 = declare_stream()
            Q_stream_c2_12 = declare_stream()
            I_stream_c1_22 = declare_stream()
            Q_stream_c1_22 = declare_stream()
            I_stream_c2_22 = declare_stream()
            Q_stream_c2_22 = declare_stream()

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

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                align()
                play('X180', qubit_1)
                update_frequency(qubit_1, self.meta_dict[qubit_1]['f_state_IF'])
                play('X180_2', qubit_1)
                update_frequency(qubit_1, self.meta_dict[qubit_1]['q_IF'])
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_20)
                save(Q1, Q_stream_c1_20)
                save(I2, I_stream_c2_20)
                save(Q2, Q_stream_c2_20)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_2)
                align()
                play('X180', qubit_2)
                update_frequency(qubit_2, self.meta_dict[qubit_2]['f_state_IF'])
                play('X180_2', qubit_2)
                update_frequency(qubit_2, self.meta_dict[qubit_2]['q_IF'])
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_02)
                save(Q1, Q_stream_c1_02)
                save(I2, I_stream_c2_02)
                save(Q2, Q_stream_c2_02)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                align()
                play('X180', qubit_1)
                play('X180', qubit_2)
                update_frequency(qubit_2, self.meta_dict[qubit_2]['f_state_IF'])
                play('X180_2', qubit_2)
                update_frequency(qubit_2, self.meta_dict[qubit_2]['q_IF'])
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_12)
                save(Q1, Q_stream_c1_12)
                save(I2, I_stream_c2_12)
                save(Q2, Q_stream_c2_12)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                align()
                play('X180', qubit_1)
                play('X180', qubit_2)
                update_frequency(qubit_1, self.meta_dict[qubit_1]['f_state_IF'])
                play('X180_2', qubit_1)
                update_frequency(qubit_1, self.meta_dict[qubit_1]['q_IF'])
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_21)
                save(Q1, Q_stream_c1_21)
                save(I2, I_stream_c2_21)
                save(Q2, Q_stream_c2_21)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_ticks, qubit_1)
                align()
                play('X180', qubit_1)
                play('X180', qubit_2)

                update_frequency(qubit_1, self.meta_dict[qubit_1]['f_state_IF'])
                play('X180_2', qubit_1)
                update_frequency(qubit_1, self.meta_dict[qubit_1]['q_IF'])

                update_frequency(qubit_2, self.meta_dict[qubit_2]['f_state_IF'])
                play('X180_2', qubit_2)
                update_frequency(qubit_2, self.meta_dict[qubit_2]['q_IF'])
                align()
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_22)
                save(Q1, Q_stream_c1_22)
                save(I2, I_stream_c2_22)
                save(Q2, Q_stream_c2_22)


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
                I_stream_c1_02.save_all('I_c1_02')
                Q_stream_c1_02.save_all('Q_c1_02')
                I_stream_c2_02.save_all('I_c2_02')
                Q_stream_c2_02.save_all('Q_c2_02')
                I_stream_c1_20.save_all('I_c1_20')
                Q_stream_c1_20.save_all('Q_c1_20')
                I_stream_c2_20.save_all('I_c2_20')
                Q_stream_c2_20.save_all('Q_c2_20')
                I_stream_c1_21.save_all('I_c1_21')
                Q_stream_c1_21.save_all('Q_c1_21')
                I_stream_c2_21.save_all('I_c2_21')
                Q_stream_c2_21.save_all('Q_c2_21')
                I_stream_c1_12.save_all('I_c1_12')
                Q_stream_c1_12.save_all('Q_c1_12')
                I_stream_c2_12.save_all('I_c2_12')
                Q_stream_c2_12.save_all('Q_c2_12')
                I_stream_c1_22.save_all('I_c1_22')
                Q_stream_c1_22.save_all('Q_c1_22')
                I_stream_c2_22.save_all('I_c2_22')
                Q_stream_c2_22.save_all('Q_c2_22')

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

        I_c1_02 = res_handles.get("I_c1_02").fetch_all()['value']
        Q_c1_02 = res_handles.get("Q_c1_02").fetch_all()['value']
        I_c2_02 = res_handles.get("I_c2_02").fetch_all()['value']
        Q_c2_02 = res_handles.get("Q_c2_02").fetch_all()['value']

        I_c1_20 = res_handles.get("I_c1_20").fetch_all()['value']
        Q_c1_20 = res_handles.get("Q_c1_20").fetch_all()['value']
        I_c2_20 = res_handles.get("I_c2_20").fetch_all()['value']
        Q_c2_20 = res_handles.get("Q_c2_20").fetch_all()['value']

        I_c1_12 = res_handles.get("I_c1_12").fetch_all()['value']
        Q_c1_12 = res_handles.get("Q_c1_12").fetch_all()['value']
        I_c2_12 = res_handles.get("I_c2_12").fetch_all()['value']
        Q_c2_12 = res_handles.get("Q_c2_12").fetch_all()['value']

        I_c1_21 = res_handles.get("I_c1_21").fetch_all()['value']
        Q_c1_21 = res_handles.get("Q_c1_21").fetch_all()['value']
        I_c2_21 = res_handles.get("I_c2_21").fetch_all()['value']
        Q_c2_21 = res_handles.get("Q_c2_21").fetch_all()['value']

        I_c1_22 = res_handles.get("I_c1_22").fetch_all()['value']
        Q_c1_22 = res_handles.get("Q_c1_22").fetch_all()['value']
        I_c2_22 = res_handles.get("I_c2_22").fetch_all()['value']
        Q_c2_22 = res_handles.get("Q_c2_22").fetch_all()['value']

        IQ_c1_00 = [np.mean(I_c1_00), np.mean(Q_c1_00)]
        IQ_c2_00 = [np.mean(I_c2_00), np.mean(Q_c2_00)]
        IQ_c1_01 = [np.mean(I_c1_01), np.mean(Q_c1_01)]
        IQ_c2_01 = [np.mean(I_c2_01), np.mean(Q_c2_01)]
        IQ_c1_10 = [np.mean(I_c1_10), np.mean(Q_c1_10)]
        IQ_c2_10 = [np.mean(I_c2_10), np.mean(Q_c2_10)]
        IQ_c1_11 = [np.mean(I_c1_11), np.mean(Q_c1_11)]
        IQ_c2_11 = [np.mean(I_c2_11), np.mean(Q_c2_11)]
        IQ_c1_02 = [np.mean(I_c1_02), np.mean(Q_c1_02)]
        IQ_c2_02 = [np.mean(I_c2_02), np.mean(Q_c2_02)]
        IQ_c1_20 = [np.mean(I_c1_20), np.mean(Q_c1_20)]
        IQ_c2_20 = [np.mean(I_c2_20), np.mean(Q_c2_20)]
        IQ_c1_12 = [np.mean(I_c1_12), np.mean(Q_c1_12)]
        IQ_c2_12 = [np.mean(I_c2_12), np.mean(Q_c2_12)]
        IQ_c1_21 = [np.mean(I_c1_21), np.mean(Q_c1_21)]
        IQ_c2_21 = [np.mean(I_c2_21), np.mean(Q_c2_21)]
        IQ_c1_22 = [np.mean(I_c1_22), np.mean(Q_c1_22)]
        IQ_c2_22 = [np.mean(I_c2_22), np.mean(Q_c2_22)]

        num_shots = self.shots
        self.confusion = np.zeros([9, 9])

        for i in range(num_shots):
            s_c1_00 = find_state([I_c1_00[i], Q_c1_00[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_00 = find_state([I_c2_00[i], Q_c2_00[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_00 + s_c2_00][0] += 1 / num_shots

            s_c1_01 = find_state([I_c1_01[i], Q_c1_01[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_01 = find_state([I_c2_01[i], Q_c2_01[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_01 + s_c2_01][1] += 1 / num_shots

            s_c1_02 = find_state([I_c1_02[i], Q_c1_02[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_02 = find_state([I_c2_02[i], Q_c2_02[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_02 + s_c2_02][2] += 1 / num_shots

            s_c1_10 = find_state([I_c1_10[i], Q_c1_10[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_10 = find_state([I_c2_10[i], Q_c2_10[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_10 + s_c2_10][3] += 1 / num_shots

            s_c1_11 = find_state([I_c1_11[i], Q_c1_11[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_11 = find_state([I_c2_11[i], Q_c2_11[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_11 + s_c2_11][4] += 1 / num_shots

            s_c1_12 = find_state([I_c1_12[i], Q_c1_12[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_12 = find_state([I_c2_12[i], Q_c2_12[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_12 + s_c2_12][5] += 1 / num_shots

            s_c1_20 = find_state([I_c1_20[i], Q_c1_20[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_20 = find_state([I_c2_20[i], Q_c2_20[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_20 + s_c2_20][6] += 1 / num_shots

            s_c1_21 = find_state([I_c1_21[i], Q_c1_21[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_21 = find_state([I_c2_21[i], Q_c2_21[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_21 + s_c2_21][7] += 1 / num_shots

            s_c1_22 = find_state([I_c1_22[i], Q_c1_22[i]], IQ_c1_00 + IQ_c1_01 + IQ_c1_02, IQ_c1_10 + IQ_c1_11 + IQ_c1_12, IQ_c1_20 + IQ_c1_21 + IQ_c1_22)
            s_c2_22 = find_state([I_c2_22[i], Q_c2_22[i]], IQ_c2_00 + IQ_c2_10 + IQ_c2_20, IQ_c2_01 + IQ_c2_11 + IQ_c2_21, IQ_c1_02 + IQ_c1_12 + IQ_c1_22)
            self.confusion[3 * s_c1_22 + s_c2_22][8] += 1 / num_shots

        self.calib_params = [IQ_c1_00, IQ_c2_00, IQ_c1_01, IQ_c2_01, IQ_c1_10, IQ_c2_10, IQ_c1_11, IQ_c2_11, IQ_c1_02,
                             IQ_c2_02, IQ_c1_20, IQ_c2_20, IQ_c1_12, IQ_c2_12, IQ_c1_21, IQ_c2_21, IQ_c1_22, IQ_c2_22]


        if self.plot:
            fig, ax = plt.subplots(2, 1, constrained_layout=True)

            ax[0].plot(Q_c1_00, I_c1_00, '.', label='00_c1')
            ax[0].plot(Q_c1_01, I_c1_01, '.', label='01_c1')
            ax[0].plot(Q_c1_02, I_c1_02, '.', label='02_c1')
            ax[0].plot(Q_c1_10, I_c1_10, '.', label='10_c1')
            ax[0].plot(Q_c1_11, I_c1_11, '.', label='11_c1')
            ax[0].plot(Q_c1_12, I_c1_12, '.', label='12_c1')
            ax[0].plot(Q_c1_20, I_c1_20, '.', label='20_c1')
            ax[0].plot(Q_c1_21, I_c1_21, '.', label='21_c1')
            ax[0].plot(Q_c1_22, I_c1_22, '.', label='22_c1')

            ax[1].plot(Q_c2_00, I_c2_00, '.', label='00_c2')
            ax[1].plot(Q_c2_01, I_c2_01, '.', label='01_c2')
            ax[1].plot(Q_c2_02, I_c2_02, '.', label='02_c2')
            ax[1].plot(Q_c2_10, I_c2_10, '.', label='10_c2')
            ax[1].plot(Q_c2_11, I_c2_11, '.', label='11_c2')
            ax[1].plot(Q_c2_12, I_c2_12, '.', label='12_c2')
            ax[1].plot(Q_c2_20, I_c2_20, '.', label='20_c2')
            ax[1].plot(Q_c2_21, I_c2_21, '.', label='21_c2')
            ax[1].plot(Q_c2_22, I_c2_22, '.', label='22_c2')

            ax[0].legend()
            ax[1].legend()
            
            ax[0].set_title(f'{self.element_1}')
            ax[1].set_title(f'{self.element_2}')

            ax[0].set_ylabel('I')
            ax[0].set_xlabel('Q')
            ax[1].set_ylabel('I')
            ax[1].set_xlabel('Q')

            fig.suptitle(f'F = {np.trace(self.confusion) / 9:.3f}')

            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

        return np.trace(self.confusion) / 2

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])