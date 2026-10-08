from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CoreLib.experiment import ExperimentClass
from Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from Helpers.fit_functions import exponential, exponential_1
from Helpers.gate_macros import flat_top_CZ

class T1(ExperimentClass):
    """
    T1 experiment
    """
    def __init__(self,element,t_min,t_max,num_t,shots,calib_params,outer_folder='',path='T1', suffix='data', meta_dict=None, plot=True, fit=True, save=False):

        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        if calib_params:
            self.t_vec_raw = np.logspace(np.log10(t_min),np.log10(t_max),num_t) #Raw time array made from input parameters
        else:
            # self.t_vec_raw = np.linspace(t_min, t_max, num_t)
            self.t_vec_raw = np.logspace(np.log10(t_min), np.log10(t_max), num_t)
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw) #Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec*4e-9 #Convert to physical time array
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.save = save

        self.job = self.acquire(self.make_prog())
        self.analyze()
        if self.save:
            self.pickle_data()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        clk_vec = self.clk_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                for t in clk_vec:
                    wait(reset_ticks, qubit)
                    align()
                    play('X180', qubit)
                    align()
                    wait(int(t), qubit)
                    # flat_top_CZ("c1", t)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def analyze(self):


        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params)
        self.data = [self.t_vec, np.mean(ss_array, 0)]

            # self.data = [self.t_vec, I_array, Q_array]

            # combined_array = A_array*np.exp(1j*phi)
            # self.data = [self.t_vec, I_array, Q_array]

        T1 = 0
        T1_err = 0

        if self.fit:
            popt, pcov = super().analyze('exponential', 'T1_seed')

            T1 = popt[2]
            T1_err = np.sqrt(pcov[2][2])

            self.T1 = T1
            self.T1_err = T1_err

            t_vec_fit = np.logspace(np.log10(self.t_vec[0]), np.log10(self.t_vec[-1]))


        if self.plot:
            # if self.calib_params:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, self.data[1], 'o')
            if self.fit:
                ax.plot(t_vec_fit, exponential(t_vec_fit, *popt))
            ax.set_title(self.element + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(
                np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            plt.semilogx()
            plt.savefig(self.iname[:-4] + 'ss' + '.png', bbox_inches="tight")
            plt.close(fig)
            # else:

####IQ

        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, self.num_t])
        I_array = np.mean(IQ_array[0], 0)
        Q_array = np.mean(IQ_array[1], 0)
        pt1 = I_array[0] + 1j * Q_array[0]
        ptf = I_array[-1] + 1j * Q_array[-1]
        phi = np.angle(ptf-pt1)
        IQ_array = (I_array + 1j*Q_array)*np.exp(-1j*phi)
        I_array = np.real(IQ_array)
        Q_array = np.imag(IQ_array)

        # amp = np.sqrt(I_array**2 + Q_array**2)
        self.data = [self.t_vec, I_array, Q_array, np.mean(ss_array, 0)]



        # I_array = I_array * np.exp(-1j*phi)
        #     Q_array = Q_array * np.exp(-1j*phi))
        T1 = 0
        T1_err = 0

        if self.fit:
            popt, pcov = super().analyze('exponential_1', 'T1_seed_1')

            T1 = popt[1]
            T1_err = np.sqrt(pcov[1][1])

            self.T1 = T1
            self.T1_err = T1_err

            t_vec_fit = np.logspace(np.log10(self.t_vec[0]), np.log10(self.t_vec[-1]))

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, self.data[1], 'o')
            if self.fit:
                ax.plot(t_vec_fit, exponential_1(t_vec_fit, *popt))
            ax.set_title(self.element + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('I_Q_data')
            plt.savefig(self.iname[:-4] + 'IQ' + '.png', bbox_inches="tight")
            plt.close(fig)

    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data['data'])