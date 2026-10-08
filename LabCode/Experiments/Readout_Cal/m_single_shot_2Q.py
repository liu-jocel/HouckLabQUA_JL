from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode.Helpers.ss_helpers import find_blob_median, find_threshold, concatenate
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class SingleShot2Q(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element_1,element_2,shots,outer_folder='',path='SingleShotCal', suffix='data',
                 confidence_threshold=6e-1, meta_dict=None, config=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config =config)

        self.element_1 = element_1
        self.element_2 = element_2
        self.confidence_threshold = confidence_threshold
        self.shots = shots
        self.plot = plot
        self.data = {
            "shots": self.shots
        }

        self.job = self.acquire(self.make_prog(), self.config)
        self.max_F = self.analyze()

    def make_prog(self):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        res_1 = self.meta_dict[self.element_1]['r_name']
        res_2 = self.meta_dict[self.element_2]['r_name']
        reset_time = self.meta_dict[self.element_1]['reset_time']
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
                wait(reset_time * u.ns, qubit_1)
                align(qubit_1, qubit_2, res_1, res_2)
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_00)
                save(Q1, Q_stream_c1_00)
                save(I2, I_stream_c2_00)
                save(Q2, Q_stream_c2_00)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit_1)
                align(qubit_1, qubit_2, res_1, res_2)
                play('X180', qubit_2)
                align(qubit_1, qubit_2, res_1, res_2)
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_01)
                save(Q1, Q_stream_c1_01)
                save(I2, I_stream_c2_01)
                save(Q2, Q_stream_c2_01)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit_1)
                align(qubit_1, qubit_2, res_1, res_2)
                play('X180', qubit_1)
                align(qubit_1, qubit_2, res_1, res_2)
                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                save(I1, I_stream_c1_10)
                save(Q1, Q_stream_c1_10)
                save(I2, I_stream_c2_10)
                save(Q2, Q_stream_c2_10)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit_1)
                align(qubit_1, qubit_2, res_1, res_2)
                play('X180', qubit_1)
                play('X180', qubit_2)
                align(qubit_1, qubit_2, res_1, res_2)
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

        c1_00 = I_c1_00 + 1j * Q_c1_00
        c1_01 = I_c1_01 + 1j * Q_c1_01
        c1_10 = I_c1_10 + 1j * Q_c1_10
        c1_11 = I_c1_11 + 1j * Q_c1_11

        c2_00 = I_c2_00 + 1j * Q_c2_00
        c2_01 = I_c2_01 + 1j * Q_c2_01
        c2_10 = I_c2_10 + 1j * Q_c2_10
        c2_11 = I_c2_11 + 1j * Q_c2_11

        num_shots = self.shots # len(c1_00)

        c1_theta = np.angle(find_blob_median(concatenate(c1_10, c1_11)) - find_blob_median(concatenate(c1_00, c1_01)))
        c2_theta = np.angle(find_blob_median(concatenate(c2_01, c2_11)) - find_blob_median(concatenate(c2_00, c2_10)))

        c1_00_rot = np.exp(-1j * c1_theta) * c1_00
        c1_01_rot = np.exp(-1j * c1_theta) * c1_01
        c1_10_rot = np.exp(-1j * c1_theta) * c1_10
        c1_11_rot = np.exp(-1j * c1_theta) * c1_11

        c2_00_rot = np.exp(-1j * c2_theta) * c2_00
        c2_01_rot = np.exp(-1j * c2_theta) * c2_01
        c2_10_rot = np.exp(-1j * c2_theta) * c2_10
        c2_11_rot = np.exp(-1j * c2_theta) * c2_11

        c1_thresh, c1_F = find_threshold(concatenate(c1_00_rot, c1_01_rot), concatenate(c1_10_rot, c1_11_rot))
        c2_thresh, c2_F = find_threshold(concatenate(c2_00_rot, c2_10_rot), concatenate(c2_01_rot, c2_11_rot))

        try:
            c1_thresh_0 = c1_thresh[np.where(c1_F > self.confidence_threshold)[0][0]]
            c2_thresh_0 = c2_thresh[np.where(c2_F > self.confidence_threshold)[0][0]]
        except:
            c1_thresh_0 = c1_thresh[np.where(c1_F > 0)[0][0]]
            c2_thresh_0 = c2_thresh[np.where(c2_F > 0)[0][0]]

        thresh_1 = c1_thresh[np.argmax(c1_F)]
        thresh_2 = c2_thresh[np.argmax(c2_F)]

        c1_0_mean = np.mean(np.real(c1_00_rot))
        c2_0_mean = np.mean(np.real(c2_00_rot))

        c1_factor = 1
        c2_factor = 1

        if c1_0_mean > thresh_1:
            c1_00_rot = -1 * c1_00_rot
            c1_01_rot = -1 * c1_01_rot
            c1_10_rot = -1 * c1_10_rot
            c1_11_rot = -1 * c1_11_rot
            thresh_1 = -1 * thresh_1
            c1_factor = -1

        if c2_0_mean > thresh_2:
            c2_00_rot = -1 * c2_00_rot
            c2_01_rot = -1 * c2_01_rot
            c2_10_rot = -1 * c2_10_rot
            c2_11_rot = -1 * c2_11_rot
            thresh_2 = -1 * thresh_2
            c2_factor = -1

        confusion = np.zeros([4, 4])

        for i in range(num_shots):
            c1_00_point = np.real(c1_00_rot[i])
            c2_00_point = np.real(c2_00_rot[i])
            if c1_00_point < thresh_1 and c2_00_point < thresh_2:
                confusion[0][0] = confusion[0][0] + 1 / num_shots
            elif (c1_00_point < thresh_1 and c2_00_point > thresh_2):
                confusion[1][0] = confusion[1][0] + 1 / num_shots
            elif (c1_00_point > thresh_1 and c2_00_point < thresh_2):
                confusion[2][0] = confusion[2][0] + 1 / num_shots
            elif (c1_00_point > thresh_1 and c2_00_point > thresh_2):
                confusion[3][0] = confusion[3][0] + 1 / num_shots

            c1_01_point = np.real(c1_01_rot[i])
            c2_01_point = np.real(c2_01_rot[i])
            if c1_01_point < thresh_1 and c2_01_point < thresh_2:
                confusion[0][1] = confusion[0][1] + 1 / num_shots
            elif (c1_01_point < thresh_1 and c2_01_point > thresh_2):
                confusion[1][1] = confusion[1][1] + 1 / num_shots
            elif (c1_01_point > thresh_1 and c2_01_point < thresh_2):
                confusion[2][1] = confusion[2][1] + 1 / num_shots
            elif (c1_01_point > thresh_1 and c2_01_point > thresh_2):
                confusion[3][1] = confusion[3][1] + 1 / num_shots

            c1_10_point = np.real(c1_10_rot[i])
            c2_10_point = np.real(c2_10_rot[i])
            if c1_10_point < thresh_1 and c2_10_point < thresh_2:
                confusion[0][2] = confusion[0][2] + 1 / num_shots
            elif (c1_10_point < thresh_1 and c2_10_point > thresh_2):
                confusion[1][2] = confusion[1][2] + 1 / num_shots
            elif (c1_10_point > thresh_1 and c2_10_point < thresh_2):
                confusion[2][2] = confusion[2][2] + 1 / num_shots
            elif (c1_10_point > thresh_1 and c2_10_point > thresh_2):
                confusion[3][2] = confusion[3][2] + 1 / num_shots

            c1_11_point = np.real(c1_11_rot[i])
            c2_11_point = np.real(c2_11_rot[i])
            if c1_11_point < thresh_1 and c2_11_point < thresh_2:
                confusion[0][3] = confusion[0][3] + 1 / num_shots
            elif (c1_11_point < thresh_1 and c2_11_point > thresh_2):
                confusion[1][3] = confusion[1][3] + 1 / num_shots
            elif (c1_11_point > thresh_1 and c2_11_point < thresh_2):
                confusion[2][3] = confusion[2][3] + 1 / num_shots
            elif (c1_11_point > thresh_1 and c2_11_point > thresh_2):
                confusion[3][3] = confusion[3][3] + 1 / num_shots

        self.confusion = confusion
        self.thresh_1 = thresh_1
        self.thresh_2 = thresh_2
        self.c1_factor = c1_factor
        self.c2_factor = c2_factor
        self.c1_theta = c1_theta + self.meta_dict[self.element_1]["read_theta"]
        self.c2_theta = c2_theta + self.meta_dict[self.element_2]["read_theta"]
        self.thresh_0_1 = c1_thresh_0
        self.thresh_0_2 = c2_thresh_0
        self.calib_params = {
            f"scale_factor_{self.element_1}": self.c1_factor,
            f"scale_factor_{self.element_2}": self.c2_factor,
            f"threshold_{self.element_1}": self.thresh_1,
            f"threshold_{self.element_2}": self.thresh_2,
            f"read_theta_{self.element_1}": self.c1_theta,
            f"read_theta_{self.element_2}": self.c2_theta,
            f"ground_threshold_{self.element_1}": self.thresh_0_1,
            f"ground_threshold_{self.element_2}": self.thresh_0_2
        }

        print("###############################################################")
        print(f"Threshold {self.element_1}: {self.thresh_1}")
        print(f"Scale Factor {self.element_1}: {self.c1_factor}")
        print(f"Theta {self.element_1}: {self.c1_theta}")
        print(f"Ground Confidence Threshold {self.element_1}: {self.thresh_0_1}")
        print("###############################################################")
        print(f"Threshold {self.element_2}: {self.thresh_2}")
        print(f"Scale Factor {self.element_2}: {self.c2_factor}")
        print(f"Theta {self.element_2}: {self.c2_theta}")
        print(f"Ground Confidence Threshold {self.element_2}: {self.thresh_0_2}")
        print("###############################################################")

        self.data.update({'calib_params': self.calib_params,
                          "c1_00": c1_00,
                          "c1_01": c1_01,
                          "c1_10": c1_10,
                          "c1_11": c1_11,
                          "c2_00": c2_00,
                          "c2_01": c2_01,
                          "c2_10": c2_10,
                          "c2_11":c2_11,
                          })
        self.fidelity = [np.max(c1_F), np.max(c2_F)]
        if self.plot:
            fig, ax = plt.subplots(2, 2,  constrained_layout=True)

            num_bins = int(np.sqrt(num_shots))

            ax[0][0].hist(np.real(c1_00_rot), bins=num_bins, density=True, histtype='step', label='00')
            ax[0][0].hist(np.real(c1_01_rot), bins=num_bins, density=True, histtype='step', label='01')
            ax[0][0].hist(np.real(c1_10_rot), bins=num_bins, density=True, histtype='step', label='10')
            ax[0][0].hist(np.real(c1_11_rot), bins=num_bins, density=True, histtype='step', label='11')
            ax[0][0].axvline(x=self.thresh_0_1, color='black', linestyle='--')
            ax[0][0].axvline(x=self.thresh_1, color='black', linestyle='-.')

            ax[0][1].hist(np.real(c2_00_rot), bins=num_bins, density=True, histtype='step', label='00')
            ax[0][1].hist(np.real(c2_01_rot), bins=num_bins, density=True, histtype='step', label='01')
            ax[0][1].hist(np.real(c2_10_rot), bins=num_bins, density=True, histtype='step', label='10')
            ax[0][1].hist(np.real(c2_11_rot), bins=num_bins, density=True, histtype='step', label='11')
            ax[0][1].axvline(x=self.thresh_0_2, color='black', linestyle='--')
            ax[0][1].axvline(x=self.thresh_2, color='black', linestyle='-.')

            ax[1][0].plot(c1_thresh, c1_F)
            ax[1][0].axvline(x=self.thresh_0_1, color='black', linestyle='--')
            ax[1][0].axvline(x=self.thresh_1, color='black', linestyle='-.')
            ax[1][1].plot(c2_thresh, c2_F)
            ax[1][1].axvline(x=self.thresh_0_2, color='black', linestyle='--')
            ax[1][1].axvline(x=self.thresh_2, color='black', linestyle='-.')

            ax[0][0].legend()
            ax[0][1].legend()

            ax[0][0].set_title('Cavity 1')
            ax[0][1].set_title('Cavity 2')

            ax[1][0].set_ylabel('Readout fidelity')
            ax[1][0].set_xlabel('Threshold')
            ax[1][1].set_xlabel('Threshold')


            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True)

            ax[0][0].scatter(np.real(c1_00_rot), np.imag(c1_00_rot), label='00')
            ax[0][0].scatter(np.real(c1_01_rot), np.imag(c1_01_rot), label='01')
            ax[0][0].scatter(np.real(c1_10_rot), np.imag(c1_10_rot), label='10')
            ax[0][0].scatter(np.real(c1_11_rot), np.imag(c1_11_rot), label='11')
            ax[0][0].set_title("Cavity 1 Rotated")
            ax[0][0].set_ylabel("Q")
            ax[0][0].set_xlabel("I")

            ax[0][1].scatter(np.real(c2_00_rot), np.imag(c2_00_rot), label='00')
            ax[0][1].scatter(np.real(c2_01_rot), np.imag(c2_01_rot), label='01')
            ax[0][1].scatter(np.real(c2_10_rot), np.imag(c2_10_rot), label='10')
            ax[0][1].scatter(np.real(c2_11_rot), np.imag(c2_11_rot), label='11')
            ax[0][1].set_title("Cavity 1 Rotated")
            ax[0][1].set_ylabel("Q")
            ax[0][1].set_xlabel("I")

            ax[1][0].scatter(np.real(c1_00), np.imag(c1_00), label='00')
            ax[1][0].scatter(np.real(c1_01), np.imag(c1_01), label='01')
            ax[1][0].scatter(np.real(c1_10), np.imag(c1_10), label='10')
            ax[1][0].scatter(np.real(c1_11), np.imag(c1_11), label='11')
            ax[1][0].set_title("Cavity 1 Rotated")
            ax[1][0].set_ylabel("Q")
            ax[1][0].set_xlabel("I")

            ax[1][1].scatter(np.real(c2_00), np.imag(c2_00), label='00')
            ax[1][1].scatter(np.real(c2_01), np.imag(c2_01), label='01')
            ax[1][1].scatter(np.real(c2_10), np.imag(c2_10), label='10')
            ax[1][1].scatter(np.real(c2_11), np.imag(c2_11), label='11')
            ax[1][1].set_title("Cavity 1 Rotated")
            ax[1][1].set_ylabel("Q")
            ax[1][1].set_xlabel("I")

            ax[0][0].legend()
            ax[0][1].legend()
            ax[1][0].legend()
            ax[1][1].legend()
            plt.suptitle("2Q Readout Blobs")
            plt.tight_layout()
            plt.savefig(self.iname[:-4] + "_blob", bbox_inches='tight')
            plt.close(fig)

        return [np.max(c1_F), np.max(c2_F)]

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])