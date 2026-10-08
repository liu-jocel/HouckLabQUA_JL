from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode2.Helpers.fit_functions import exponential

class Rabi_2Q(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self,element_1,element_2,a_min,a_max,num_a,shots,calib_params,outer_folder='',path='Rabi', suffix='data', cfg=None,confusion=np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, cfg=cfg)

        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.pi_amp_norm = element_1['pi_amp']
        self.a_min = a_min
        self.a_max = a_max
        self.num_a = num_a
        self.amp_vec = np.linspace(self.a_min, self.a_max, self.num_a) #Raw amplitude array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion

        self.acquire()
        self.analyze()

    def make_prog(self):

        qubit = self.element_2['q_name']
        reset_ticks = self.element_1['reset_ticks']
        amp_vec = self.amp_vec/self.pi_amp_norm #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            a = declare(int)
            amps = declare(fixed,value=amp_vec)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                # for a in amp_vec:
                with for_(a, 0, a < len(amp_vec), a + 1):
                    amplitude = amps[a]
                    wait(reset_ticks, qubit)
                    play('X180'*amp(amplitude), qubit)
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.element_1, self.element_2, I1, Q1, I2, Q2)

                    save(I1, I1_stream)
                    save(Q1, Q1_stream)
                    save(I2, I2_stream)
                    save(Q2, Q2_stream)

            with stream_processing():
                I1_stream.save_all("I_c1")
                Q1_stream.save_all("Q_c1")
                I2_stream.save_all("I_c2")
                Q2_stream.save_all("Q_c2")

        return prog

    def acquire(self):

        self.job = super().acquire(self.make_prog())

    def simulate(self, channels, ticks, ticks_to_plot):

        self.samples = super().simulate(self.make_prog(), channels, ticks, ticks_to_plot)

    def analyze(self):

        counts_array = super().job_to_shots_2q(self.job, [self.shots, self.num_a], self.calib_params)
        counts_array = np.linalg.inv(self.confusion)@counts_array
        self.data = [self.amp_vec, counts_array]

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(self.amp_vec, counts_array[0]/self.shots, 'o', label='00')
        ax.plot(self.amp_vec, counts_array[1]/self.shots, 'o', label='01')
        ax.plot(self.amp_vec, counts_array[2]/self.shots, 'o', label='10')
        ax.plot(self.amp_vec, counts_array[3]/self.shots, 'o', label='11')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('State probability')
        ax.set_title('Joint Rabi')
        ax.legend()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)



    def analyze_old(self):

        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, self.num_a])
        I_array = np.mean(IQ_array[0],0)
        Q_array = np.mean(IQ_array[1],0)
        self.data = [self.amp_vec, I_array, Q_array]

        # ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params)
        # self.data = [self.t_vec, np.mean(ss_array, 0)]
        #
        # popt, pcov = super().analyze('exponential', 'T1_seed')
        #
        # T1 = popt[2]
        # T1_err = np.sqrt(pcov[2][2])
        #
        # t_vec_fit = np.logspace(np.log10(self.t_vec[0]), np.log10(self.t_vec[-1]))
        #
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(self.amp_vec, self.data[1]-np.mean(self.data[1]), 'o')
        ax.plot(self.amp_vec, self.data[2]-np.mean(self.data[2]), 'o')
        ax.set_xlabel('Amp (arb.)')
        #ax.set_ylabel('Excited state probability')
        #ax.set_title(self.element['q_name']+', T1 = ' + T1 + '+/-' + str(np.round(T1_err,4)) + ' s')
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])