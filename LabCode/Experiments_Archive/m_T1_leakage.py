from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.SPAM_correction import correct_counts_array

class T1_leakage(ExperimentClass):
    """
    T1 leakage experiment
    """
    def __init__(self,element,t_min,t_max,num_t,shots,calib_params,confusion=np.eye(3),outer_folder='',path='T1', suffix='data', meta_dict=None, plot=True, fit=True, save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        self.t_vec_raw = np.linspace(t_min,t_max,num_t) #Raw time array made from input parameters
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw) #Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec*4e-9 #Convert to physical time array
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.save = save
        self.confusion=confusion

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
                    play('X180', qubit)
                    wait(int(t), qubit)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def analyze(self):

        ss_array = super().job_to_shots_1q_3state(self.job, [self.shots, self.num_t], self.calib_params)

        counts_array_raw = super().ss_array_to_counts(ss_array) # counts before correction
        counts_array = correct_counts_array(self.confusion,counts_array_raw) # counts after correction

        self.data = {'time':self.t_vec,
                     'raw_probs': [counts_array_raw[0]/self.shots, counts_array_raw[1]/self.shots, counts_array_raw[2]/self.shots],
                     'confusion': self.confusion,
                     'corrected_probs': [counts_array[0]/self.shots, counts_array[1]/self.shots, counts_array[2]/self.shots]}

        T1 = 0
        T1_err = 0

        if self.fit:
            pass
            # popt, pcov = super().analyze('exponential', 'T1_seed')
            #
            # T1 = popt[2]
            # T1_err = np.sqrt(pcov[2][2])
            #
            # self.T1 = T1
            # self.T1_err = T1_err
            #
            # t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1])


        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, counts_array[0]/self.shots, 'o',label='|0>')
            ax.plot(self.t_vec, counts_array[1]/self.shots, 'o',label='|1>')
            ax.plot(self.t_vec, 10*counts_array[2]/self.shots, 'o',label='|2> (x10)')
            if self.fit:
                pass
                # ax.plot(t_vec_fit, exponential(t_vec_fit, *popt))
            ax.set_title(self.element + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(
                np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data['data'])