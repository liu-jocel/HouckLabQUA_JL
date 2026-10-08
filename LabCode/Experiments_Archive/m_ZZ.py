from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode.Helpers.fit_functions import decaying_cosine, cosine

class ZZ(ExperimentClass):
    """
    T2R experiement
    """
    def __init__(self,element_1,element_2,t_min,t_max,num_t,shots,detuning,calib_params,outer_folder='',path='ZZ', suffix='data', meta_dict=None, plot=True, fit=True, save = True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element_1 = element_1
        self.element_2 = element_2 #meta dict element
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        self.t_vec_raw = np.linspace(t_min,t_max,num_t) #Raw time array made from input parameters
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw) #Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec*4e-9 #Convert to physical time array
        self.shots = shots
        self.detuning = detuning
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.save = save

        self.job = self.acquire(self.make_prog())
        self.analyze()

        if self.save:
            self.pickle_data()


    def make_prog(self):

        qubit = self.meta_dict[self.element_1]['q_name']
        spectator = self.meta_dict[self.element_2]['q_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        clk_vec = self.clk_vec
        shots = self.shots
        detuned_freq = self.meta_dict[self.element_1]['q_IF'] + self.detuning

        with program() as prog:
            n = declare(int)
            c = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            clks = declare(int, value=clk_vec)
            I_stream_nopi = declare_stream()
            Q_stream_nopi = declare_stream()
            I_stream_pi = declare_stream()
            Q_stream_pi = declare_stream()
            update_frequency(qubit, detuned_freq)

            with for_(n, 0, n < shots, n + 1):
                with for_(c, 0, c< len(clk_vec), c+1):
                    clk = clks[c]
                    wait(reset_ticks, qubit)
                    align()
                    play('X90', qubit)
                    wait(clk, qubit)
                    play('X90', qubit)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.element_1], I, Q)

                    save(I, I_stream_nopi)
                    save(Q, Q_stream_nopi)

                    align()

                    wait(reset_ticks, qubit)
                    align()
                    play('X180', spectator)
                    align(qubit, spectator)
                    play('X90', qubit)
                    wait(clk, qubit)
                    play('X90', qubit)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.element_1], I, Q)

                    save(I, I_stream_pi)
                    save(Q, Q_stream_pi)

            with stream_processing():
                I_stream_nopi.save_all("I_nopi")
                Q_stream_nopi.save_all("Q_nopi")
                I_stream_pi.save_all("I_pi")
                Q_stream_pi.save_all("Q_pi")

        return prog

    def analyze(self):

        ss_array_nopi = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params, I_name='I_nopi',
                                                Q_name='Q_nopi')
        ss_array_pi = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params, I_name='I_pi',
                                              Q_name='Q_pi')

        self.ZZ = 0

        if self.fit:
            self.data = [self.t_vec,np.mean(ss_array_nopi,0)]
            popt_nopi, pcov_nopi = super().analyze('cosine', 'freq_cal_seed', detuning = self.detuning)
            self.data = [self.t_vec, np.mean(ss_array_pi,0)]
            popt_pi, pcov_pi = super().analyze('cosine', 'freq_cal_seed', detuning=self.detuning)

            self.freq_nopi = popt_nopi[1]
            self.freq_pi = popt_pi[1]
            self.ZZ = self.freq_nopi - self.freq_pi
            self.ZZ_err = np.sqrt(pcov_nopi[1][1] + pcov_pi[1][1])

            t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1])
        else:
            self.data = [self.t_vec,np.mean(ss_array_nopi,0),np.mean(ss_array_pi,0)]

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, np.mean(ss_array_nopi,0), 'o')
            ax.plot(self.t_vec, np.mean(ss_array_pi, 0), 'o')
            if self.fit:
                ax.plot(t_vec_fit, cosine(t_vec_fit, *popt_nopi))
                ax.plot(t_vec_fit, cosine(t_vec_fit, *popt_pi))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            ax.set_title(self.element_1+', ZZ = ' + str(np.round(self.ZZ/1e6,4)) + ' MHz')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])