from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q, measure_3q
from LabCode2.Helpers.fit_functions import decaying_cosine
from LabCode2.Helpers.gate_macros import flat_top_CZ, flat_top_iSWAP
from LabCode2.Helpers.SPAM_correction import correct_counts_array
class Rabi_arb(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,element,a_min,a_max,num_a,shots,discrim_class,outer_folder='',path='SWAP', suffix='data', meta_dict=None, plot=True, fit=False, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.discrim = discrim_class
        self.element = element #meta dict element
        self.pi_amp_norm = self.meta_dict[self.element]['pi_amp']
        self.a_min = a_min
        self.a_max = a_max
        self.num_a = num_a
        self.amp_vec = np.linspace(a_min, a_max, num_a)  # Raw time array made from input parameters
        self.shots = shots
        # self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.confusion = confusion

        self.job = self.acquire(self.make_prog())
        self.analyze()


    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        amp_vec = self.amp_vec/self.pi_amp_norm #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            Ic = declare(fixed)
            Qc = declare(fixed)

            a = declare(int)
            amps = declare(fixed, value=amp_vec)

            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()
            Ic_stream = declare_stream()
            Qc_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(a, 0, a < len(amp_vec), a + 1):
                    amplitude = amps[a]
                    wait(reset_ticks, qubit)
                    play('X180' * amp(amplitude), qubit)
                    align()
                    I1, Q1, I2, Q2, Ic, Qc = measure_3q(self.meta_dict["q1"], self.meta_dict["q2"],
                                                        self.meta_dict["qc"],
                                                        I1, Q1, I2, Q2, Ic, Qc)

                    save(I1, I1_stream)
                    save(Q1, Q1_stream)
                    save(I2, I2_stream)
                    save(Q2, Q2_stream)
                    save(Ic, Ic_stream)  #
                    save(Qc, Qc_stream)  #

            with stream_processing():
                I1_stream.save_all("I1")
                Q1_stream.save_all("Q1")
                I2_stream.save_all("I2")
                Q2_stream.save_all("Q2")
                Ic_stream.save_all("Ic")
                Qc_stream.save_all("Qc")

        return prog

    def analyze(self):


        counts_dict = super().job_to_shots_arb(self.job, [self.shots, self.num_a], self.discrim)
        # counts_array = np.linalg.inv(self.confusion)@counts_array
        counts_array = np.zeros((len(counts_dict), self.num_a))
        for j, key in enumerate(self.discrim.POVM.keys()):
            counts_array[j, :] = counts_dict[key]

        # counts_array = correct_counts_array(self.confusion,counts_array)

        self.data = [self.amp_vec, counts_array, list(self.discrim.POVM.keys())]

        if self.fit:
            try:
                popt, pcov = super().analyze('decaying_cosine', 'rabi_seed', ydata=counts_array[1])

                if popt[0] < 0:
                    popt[0] = popt[0] * -1
                    popt[2] = popt[2] + 0.5

                self.pulse_time = 1/(2*popt[1])
                flag = True
            except:
                self.pulse_time= 0
                flag = False
        else:
            self.pulse_time = 0

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)

            for c,state in enumerate(self.discrim.POVM.keys()):
                ax.plot(self.amp_vec, counts_array[c], 'o', label=state)

            if self.fit and flag:
                ax.plot(self.amp_vec, decaying_cosine(self.amp_vec, *popt), label='fit')
            # ax.axvline(x=self.pulse_time)
            ax.set_xlabel('Amp')
            ax.set_ylabel('State probability')
            ax.set_title('Rabi Oscillation, coupler freq = ' + str(self.meta_dict['c1']['c_LO']['LO_freq']/1e9 +
                                                                   self.meta_dict['c1']['c_IF']/1e9)
                         +  ', amp = ' + str(self.meta_dict['c1']['pulse_amp']) + ', Rabi amp = ' + str(self.pulse_time))
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])