from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q, measure_2q
from LabCode2.Helpers.fit_functions import exponential, exponential_1
from LabCode2.Helpers.gate_macros import flat_top_CZ

class T2R_2Q(ExperimentClass):
    """
    T1 experiment
    """
    def __init__(self,element_1, element_2, t_min,t_max,num_t,shots,calib_params,outer_folder='',path='T1', suffix='data', meta_dict=None, plot=True, fit=True, save=False, sscor = True):

        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        if sscor == True:
            self.t_vec_raw = np.logspace(np.log10(t_min), np.log10(t_max), num_t)
        else:
        # if calib_params:
            self.t_vec_raw = np.logspace(np.log10(t_min),np.log10(t_max),num_t) #Raw time array made from input parameters
        # else:
        #     # self.t_vec_raw = np.linspace(t_min, t_max, num_t)
        #     self.t_vec_raw = np.logspace(np.log10(t_min), np.log10(t_max), num_t)
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw) #Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec*4e-9 #Convert to physical time array
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.save = save
        self.sscor = sscor

        self.job = self.acquire(self.make_prog())
        self.analyze()
        if self.save:
            self.pickle_data()

    def make_prog(self):

        qubit1 = self.meta_dict[self.element_1]['q_name']
        qubit2 = self.meta_dict[self.element_2]['q_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        clk_vec = self.clk_vec
        shots = self.shots

        start = time.time_ns()

        with program() as prog:

            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            # prev_time = declare(int)
            # time_diff = declare(int)

            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()
            I_stream_2 = declare_stream()
            Q_stream_2 = declare_stream()
            # time_diff_stream = declare_stream()

            # prev_time = time.process_time_ns()

            with for_(n, 0, n < shots, n + 1):
                for t in clk_vec:

                    wait(reset_ticks, qubit1)
                    align()
                    play('X90', qubit1)
                    align()
                    play('X90', qubit2)
                    align()
                    wait(int(t), qubit1, qubit2)
                    # wait(int(t), qubit2)# maybe not required
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                    # time_diff = time.process_time_ns() - prev_time

                    save(I1, I_stream_1)
                    save(Q1, Q_stream_1)
                    save(I2, I_stream_2)
                    save(Q2, Q_stream_2)
                    # save(time_diff, time_diff_stream)

            with stream_processing():
                I_stream_1.save_all('I1')
                Q_stream_1.save_all('Q1')
                I_stream_2.save_all('I2')
                Q_stream_2.save_all('Q2')
                # time_diff_stream.save_all("time_diffs")

        return prog

    def analyze(self):

        # time_diffs = self.job.result_handles.get("time_diffs").fetch_all()

        ss_array = super().job_to_shots_2q_IQ(self.job, [self.shots, self.num_t], self.calib_params)

        if self.sscor == True:
            self.data = [self.t_vec, ss_array[0], ss_array[1]]#, time_diffs]

        else:
            self.data = [self.t_vec, np.mean(ss_array[0], 0), np.mean(ss_array[1], 0)]



            # self.data = [self.t_vec, I_array, Q_array]

            # combined_array = A_array*np.exp(1j*phi)
            # self.data = [self.t_vec, I_array, Q_array]
        #for qubit 1
        # self.data = [self.t_vec, np.mean(ss_array[0], 0), np.mean(ss_array[1], 0)]
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
            ax.set_title(self.element_1 + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(
                np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            plt.semilogx()
            plt.savefig(self.iname[:-4] + self.element_1 + '.png', bbox_inches="tight")
            plt.close(fig)
            # else:

            # for qubit 2
        # self.data = [self.t_vec, np.mean(ss_array[1], 0), np.mean(ss_array[0], 0)]
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
            ax.set_title(self.element_2 + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(
                np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            plt.semilogx()
            plt.savefig(self.iname[:-4] + self.element_2 + '.png', bbox_inches="tight")
            plt.close(fig)

        # self.data = [self.t_vec, np.mean(ss_array[0], 0), np.mean(ss_array[1], 0)]
                # else:

# ####IQ
#
#         IQ_array = super().job_to_IQ_1q(self.job, [self.shots, self.num_t])
#         I_array = np.mean(IQ_array[0], 0)
#         Q_array = np.mean(IQ_array[1], 0)
#         pt1 = I_array[0] + 1j * Q_array[0]
#         ptf = I_array[-1] + 1j * Q_array[-1]
#         phi = np.angle(ptf-pt1)
#         IQ_array = (I_array + 1j*Q_array)*np.exp(-1j*phi)
#         I_array = np.real(IQ_array)
#         Q_array = np.imag(IQ_array)
#
#         # amp = np.sqrt(I_array**2 + Q_array**2)
#         self.data = [self.t_vec, I_array, Q_array, np.mean(ss_array, 0)]
#
#
#
#         # I_array = I_array * np.exp(-1j*phi)
#         #     Q_array = Q_array * np.exp(-1j*phi))
#         T1 = 0
#         T1_err = 0
#
#         if self.fit:
#             popt, pcov = super().analyze('exponential_1', 'T1_seed_1')
#
#             T1 = popt[1]
#             T1_err = np.sqrt(pcov[1][1])
#
#             self.T1 = T1
#             self.T1_err = T1_err
#
#             t_vec_fit = np.logspace(np.log10(self.t_vec[0]), np.log10(self.t_vec[-1]))
#
#         if self.plot:
#             fig, ax = plt.subplots(1, 1, constrained_layout=True)
#             ax.plot(self.t_vec, self.data[1], 'o')
#             if self.fit:
#                 ax.plot(t_vec_fit, exponential_1(t_vec_fit, *popt))
#             ax.set_title(self.element + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(np.round(T1_err * 1e6, 4)) + ' us')
#             ax.set_xlabel('Time (s)')
#             ax.set_ylabel('I_Q_data')
#             plt.savefig(self.iname[:-4] + 'IQ' + '.png', bbox_inches="tight")
#             plt.close(fig)
#
#     # def save_data(self, data=None):
#     #     print(f'Saving {self.fname}')
#     #     super().save_data(data=data['data'])