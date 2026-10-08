import pickle
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from qm.qua import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from qualang_tools.loops import from_array
from LabCode2.Helpers.fit_functions import decaying_cosine
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class FreqAmpTune(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self,element, a_span,num_a, f_span, num_f, num_pi, shots, pulse_type="X180",
                 outer_folder='',path='FreqAmpTune', suffix='data', meta_dict=None,plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element
        self.pulse_type = pulse_type
        self.pi_amp_norm = self.meta_dict[self.element]['pi_amp']
        self.pi2_amp_norm = self.meta_dict[self.element]['pi2_amp']

        self.a_span = a_span
        self.num_a = num_a
        self.f_span = f_span
        self.num_f = num_f
        self.num_pi = num_pi
        self.amp_vec = np.linspace(1-self.a_span/2, 1+self.a_span/2, self.num_a) #Raw amplitude array made from input parameters
        self.freq_vec = np.linspace(-self.f_span/2, self.f_span/2, self.num_f).astype(int) #Raw amplitude array made from input parameters
        self.shots = shots
        self.plot = plot

        self.job = self.acquire(self.make_prog())
        self.analyze()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        amp_vec = self.amp_vec
        num_pi = self.num_pi
        f_vec = self.freq_vec
        shots = self.shots

        if self.pulse_type == 'X180':
            qubit_IF = self.meta_dict[self.element]["q_IF"]
            with program() as prog:
                n = declare(int)
                m = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                a = declare(fixed)
                df = declare(int)
                I_stream = declare_stream()
                Q_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_each_((df), (f_vec)):
                        update_frequency(qubit, df + qubit_IF)
                        with for_(*from_array(a, amp_vec)):
                            wait(reset_ticks, qubit)
                            with for_(m, 0, m < num_pi, m + 1):
                                play('X180' * amp(a), qubit)
                                play('X180' * amp(a), qubit)
                            I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.save_all("I")
                    Q_stream.save_all("Q")

        elif self.pulse_type == 'X90':
            qubit_IF = self.meta_dict[self.element]["q_pi2_IF"]
            with program() as prog:
                n = declare(int)
                m = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                a = declare(fixed)
                df = declare(int)
                I_stream = declare_stream()
                Q_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_each_((df), (f_vec)):
                        update_frequency(qubit, df + qubit_IF)
                        with for_(*from_array(a, amp_vec)):
                            wait(reset_ticks, qubit)
                            with for_(m, 0, m < num_pi, m + 1):
                                play('Y90' * amp(a), qubit)
                                play('Y90' * amp(a), qubit)
                                play('Y90' * amp(a), qubit)
                                play('Y90' * amp(a), qubit)
                            I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.save_all("I")
                    Q_stream.save_all("Q")

        else:
            print('Invalid Pulse')

        return prog

    def analyze(self):
        if self.pulse_type == 'X180':
            amp_norm = self.pi_amp_norm
            qubit_IF = self.meta_dict[self.element]["q_IF"]
        else:
            amp_norm = self.pi2_amp_norm
            qubit_IF = self.meta_dict[self.element]["q_pi2_IF"]
        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, self.num_f, self.num_a])
        I_array = np.mean(IQ_array[0], 0)
        Q_array = np.mean(IQ_array[1], 0)
        self.data = dict(freq=self.freq_vec, amp=self.amp_vec, I=I_array, Q=Q_array)
        # self.data["Magnitude"] = np.sqrt(I_array**2 + Q_array**2)
        min_I_index = np.argmax(I_array)
        freq_index, amp_index = np.unravel_index(min_I_index, I_array.shape)
        if self.plot:
            fig, ax = plt.subplots(2, 1, constrained_layout=True)

            meshI = ax[0].pcolormesh(self.amp_vec*amp_norm, self.freq_vec/u.MHz, I_array)
            ax[0].set_ylabel('Detuning (MHz)')
            ax[0].set_xlabel('Amplitude')
            ax[0].set_title("I Data")

            meshQ = ax[1].pcolormesh(self.amp_vec*amp_norm, self.freq_vec/u.MHz, Q_array)
            ax[1].set_ylabel('Detuning (MHz)')
            ax[1].set_xlabel('Amplitude')
            ax[1].set_title("Q Data:")

            fig.suptitle(f'# 2Pi Pulses: {self.num_pi}, Detuning: {self.freq_vec[freq_index]/u.MHz}, Amplitude: {self.amp_vec[amp_index]*amp_norm}')
            plt.colorbar(meshI)
            plt.colorbar(meshQ)
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)
        print("Frequency Detuning (MHz):")
        print(qubit_IF + self.freq_vec[freq_index]/u.MHz)
        print("Amplitude:")
        print(self.amp_vec[amp_index]*amp_norm)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])

if __name__ == "__main__":
    outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
    pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

    # Get metaDict
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)
    q = "q4"
    Q = meta_dict[q]
    LOs = [Q['q_LO'], Q['r_LO']]#, Q2['q_LO'], Q2['r_LO'], Qc['q_LO'], Qc['r_LO']]
    # set_LO_parameters(LOs)
    a_span = .05
    num_a = 20 #20
    f_span = 5000 * u.kHz
    num_f = 20 # 200
    num_pi = 10
    shots = 50

    exp = FreqAmpTune(q,a_span, num_a, f_span, num_f, num_pi, shots, pulse_type="X180", outer_folder=outer_folder, path="tuning",
                      suffix="Freq_Amp_Tune", meta_dict=meta_dict, plot=True,)
    exp.analyze()
    # exp.save_data()