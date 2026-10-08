from qm.qua import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import find_blob_mean, find_threshold, find_blob_median
from scipy.stats import norm
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
class SingleShot1Q(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self, element, shots, outer_folder='', path='SingleShotCal', suffix='data', meta_dict=None, config=None, save=True,
                 plot=True, repeats=1, confidence_threshold=6e-1, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

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
            m = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_0)
                save(Q, Q_stream_0)
                assign(I_reset, I)
                align(qubit, resonator)

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, qubit)
                with for_(m, 0, m < self.repeats, m+1):
                    play('X180', qubit)
                    #play('X90', qubit)
                align(qubit, resonator)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_stream_1)
                save(Q, Q_stream_1)
                assign(I_reset, I)
                align(qubit, resonator)


            with stream_processing():
                I_stream_0.save_all("I_0")
                Q_stream_0.save_all("Q_0")
                I_stream_1.save_all("I_1")
                Q_stream_1.save_all("Q_1")

        return prog

    def analyze(self):
        job = self.job

        # Extract Raw data
        I_handle_0 = job.result_handles.get("I_0")
        Q_handle_0 = job.result_handles.get("Q_0")
        I_handle_1 = job.result_handles.get("I_1")
        Q_handle_1 = job.result_handles.get("Q_1")

        I_handle_0.wait_for_all_values()
        Q_handle_0.wait_for_all_values()
        I_handle_1.wait_for_all_values()
        Q_handle_1.wait_for_all_values()

        I_0 = I_handle_0.fetch_all()['value']
        Q_0 = Q_handle_0.fetch_all()['value']
        I_1 = I_handle_1.fetch_all()['value']
        Q_1 = Q_handle_1.fetch_all()['value']

        # combine IQ data
        c_0 = I_0 + 1j * Q_0
        c_1 = I_1 + 1j * Q_1

        self.data.update({'cs': np.array([c_0, c_1])})

        num_shots = self.shots

        # Calculate Rotation Angle
        c_theta = np.angle(find_blob_median(c_1) - find_blob_median(c_0))

        c_0_rot = np.exp(-1j * c_theta) * c_0
        c_1_rot = np.exp(-1j * c_theta) * c_1

        c_thresh, c_F = find_threshold(c_0_rot, c_1_rot)

        thresh = c_thresh[np.argmax(c_F)]
        try:
            thresh_0 = c_thresh[np.where(c_F > self.confidence_threshold)[0][0]]
        except:
            thresh_0 = c_thresh[np.where(c_F > 0)[0][0]]

        c_0_mean = np.mean(np.real(c_0_rot))

        c_factor = 1

        if c_0_mean > thresh:
            c_0_rot = -1 * c_0_rot
            c_1_rot = -1 * c_1_rot
            thresh = -1 * thresh
            c_factor = -1

        confusion = np.zeros([2, 2])

        for i in range(num_shots):
            c_0_point = np.real(c_0_rot[i])
            if c_0_point < thresh:
                confusion[0][0] = confusion[0][0] + 1 / num_shots
            else:
                confusion[1][0] = confusion[1][0] + 1 / num_shots

            c_1_point = np.real(c_1_rot[i])
            if c_1_point < thresh:
                confusion[0][1] = confusion[0][1] + 1 / num_shots
            else:
                confusion[1][1] = confusion[1][1] + 1 / num_shots

        self.confusion = confusion
        self.thresh = thresh
        self.c_factor = c_factor
        self.c_theta = c_theta + self.meta_dict["read_theta"]
        self.calib_params = {
            "scale_factor": self.c_factor,
            "threshold": self.thresh,
            "read_theta": self.c_theta,
            "ground_threshold": thresh_0,
            "confusion": confusion,
        }

        print("###################################################")
        print(f"Threshold: {self.thresh}")
        print(f"Scale Factor: {self.c_factor}")
        print(f"Theta: {self.c_theta}")
        print(f"Ground Confidence Threshold: {thresh_0}")
        print(f"Fidelity {np.max(c_F)}")
        print(f"Confusion Matrix: {self.confusion}")
        print("###################################################")

        self.data.update({'calib_params': self.calib_params, "confusion": self.confusion})
        num_shots = len(c_0_rot)
        num_bins = int(np.sqrt(num_shots))
        if self.plot:
            real_c_0_rot = np.real(c_0_rot)
            real_c_1_rot = np.real(c_1_rot)
            imag_c_0_rot = np.imag(c_0_rot)
            imag_c_1_rot = np.imag(c_1_rot)

            fig, ax = plt.subplots(2, 1, constrained_layout=True)

            ax[0].hist(np.real(c_0_rot), bins=num_bins, density=True, histtype='step', color='black', label='0')
            ax[0].hist(np.real(c_1_rot), bins=num_bins, density=True, histtype='step', label='1', color = 'red')
            ax[0].axvline(x=thresh_0, color='blue', linestyle='--')
            ax[0].axvline(x=self.thresh, color='blue', linestyle='-.')

            ax[1].plot(c_thresh, c_F)

            ax[0].legend()

            ax[0].set_title('Cavity ' + self.element[-1] + f' F = {np.max(c_F)}')

            ax[1].set_ylabel('Readout infidelity')
            ax[1].set_xlabel('Threshold')

            plt.savefig(self.iname[:-4]+".jpg", bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(2, figsize=(5,10))
            ax[0].scatter(real_c_0_rot, imag_c_0_rot, label="0", color="red", alpha=1)
            ax[0].scatter(real_c_1_rot, imag_c_1_rot, label="1", color="black", alpha=0.1)
            ax[0].set_title("Rotated Data")
            ax[1].scatter(I_0, Q_0, label="0", color="red", alpha=1)
            ax[1].scatter(I_1, Q_1, label="1", color="black", alpha=0.1)
            ax[1].scatter([np.real(find_blob_median(c_1)), np.real(find_blob_median(c_0))],
                          [np.imag(find_blob_median(c_1)), np.imag(find_blob_median(c_0))], color="cyan",
                          marker="x", s=100, linewidths=2)
            # Adjust the axis limits to ensure equal aspect ratio
            ax[1].set_title("Original Data")
            plt.ylabel('Q')
            plt.xlabel('I')
            ax[0].legend()
            ax[1].legend()
            plt.suptitle(f"SS CAL, readout repeated {self.repeats} times.")
            plt.tight_layout()
            plt.savefig(self.iname[:-4]+"_blob.jpg", bbox_inches='tight')
            plt.close(fig)

        self.qm.close()
        return np.max(c_F)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])