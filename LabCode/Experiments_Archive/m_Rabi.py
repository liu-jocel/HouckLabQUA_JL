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
class Rabi(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self,element,a_min,a_max,num_a,shots,calib_params, num_pi_pulses=1, outer_folder='', path='Rabi', suffix='data',
                 meta_dict=None, ss=False,plot=True,  pulse_type="X180"):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element name
        self.pulse_type = pulse_type
        if pulse_type == "X180":
            self.pi_amp_norm = self.meta_dict[self.element]['pi_amp']
        if pulse_type == "X90":
            self.pi_amp_norm = self.meta_dict[self.element]['pi2_amp']

        self.num_pi = num_pi_pulses
        self.a_min = a_min
        self.a_max = a_max
        self.num_a = num_a
        self.amp_vec = np.linspace(self.a_min, self.a_max, self.num_a) #Raw amplitude array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.ss = ss
        self.plot = plot
        self.read_len = self.meta_dict['global_params']['read_len']

        self.job = self.acquire(self.make_prog())

        if ss:
            self.analyze_ss()
        else:
            self.analyze()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        amp_vec = self.amp_vec   # #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            a = declare(fixed)
            m = declare(int)
            # amps = declare(fixed,value=amp_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(a, amp_vec)):
                    wait(reset_ticks, qubit)
                    with for_(m, 0, m < self.num_pi, m + 1):
                        # reset_frame(qubit)
                        # reset_phase(qubit)
                        # play("X180", "q4")
                        update_frequency(qubit, -int(180*u.MHz), keep_phase=True)
                        play(self.pulse_type*amp(a), qubit)
                    I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def analyze(self):

        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, self.num_a])
        I_array = np.mean(IQ_array[0], 0)
        Q_array = np.mean(IQ_array[1], 0)
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
        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.amp_vec*self.pi_amp_norm, self.data[1] - np.mean(self.data[1]), 'o')
            ax.plot(self.amp_vec*self.pi_amp_norm, self.data[2] - np.mean(self.data[2]), 'o')
            ax.set_xlabel('Amp (arb.)')
            plt.grid()
            #ax.set_ylabel('Excited state probability')
            #ax.set_title(self.element['q_name']+', T1 = ' + T1 + '+/-' + str(np.round(T1_err,4)) + ' s')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def analyze_ss(self):

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_a], self.calib_params)
        self.data = [self.amp_vec, np.mean(ss_array, 0)]

        # popt, pcov = super().analyze('decaying_cosine', 'rabi_seed')
        #
        # if popt[0] < 0:
        #     popt[0] = popt[0] * -1
        #     popt[2] = popt[2] + 0.5
        #
        # self.pulse_amp = 0.5/popt[1]

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.amp_vec*self.pi_amp_norm, self.data[1], 'o')
            # ax.plot(self.amp_vec, decaying_cosine(self.amp_vec,*popt))
            ax.set_xlabel('Amp (V)')
            ax.set_ylabel('Excited state probability')
            plt.grid()
            # ax.set_title(self.element+', Pi amp = ' + str(np.round(self.pulse_amp,4)))
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])