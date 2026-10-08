from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import exponential
from qualang_tools.loops import from_array
from qm import simulate

class T2CPMG(ExperimentClass):
    """
    T2CPMG experiement
    """
    def __init__(self, element, t_min, t_max, num_t, numpi, shots, calib_params, outer_folder='', path='T2R', suffix='data', meta_dict=None, plot=True, fit=True, save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element
        self.t_min = t_min #
        self.t_max = t_max
        self.num_t = num_t
        self.t_vec_raw = np.linspace(t_min,t_max,num_t)/2 #Raw time array made from input parameters. Wait time is cut in half in qua
        self.clk_tvec = 2*time_vec_to_clk_vec_wait(self.t_vec_raw) #Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_tvec*4e-9 #Convert to physical time array
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.save = save
        self.numpi = numpi # total pi pulses in the program

        # self.simulate(self.make_prog, ['3', '4'],140000,[[100,500000]])
        self.job = self.acquire(self.make_prog())
        self.analyze()

        if self.save:
            self.pickle_data()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        clk_tvec = self.clk_tvec
        shots = self.shots
        numpi = self.numpi
        with program() as prog:
            n = declare(int)
            t = declare(int)
            n_pi = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                # with for_(t, 0, t < len(clk_tvec), t + 1):
                #     tau = t/numpi
                #     #replace the following with N pi pulses. Gap between
                #     wait(reset_ticks, qubit)
                #     play('X90', qubit)
                #     wait(tau/2, qubit)
                #     for _ in range(numpi-1):
                #         play('Y180', qubit)
                #         wait(tau, qubit)
                #     play('Y180', qubit)
                #     wait(tau/2, qubit)
                #     play('Xm90', qubit)
                #     align()
                #     I, Q = measure_1q(self.meta_dict[self.element], I, Q)
                #
                #     save(I, I_stream)
                #     save(Q, Q_stream)

                # for t in clk_tvec:
                with for_each_(t, clk_tvec):
                        tau = t/numpi
                        #replace the following with N pi pulses. Gap between
                        wait(reset_ticks, qubit)
                        play('X90', qubit)
                        wait(tau/2, qubit)
                        with for_(n_pi, 0, n_pi < numpi-1, n_pi + 1):
                            play('Y180', qubit)
                            wait(tau, qubit)
                        play('Y180', qubit)
                        wait(tau/2, qubit)
                        play('Xm90', qubit)
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

        T2 = 0
        T2_err = 0

        if self.fit:
            popt, pcov = super().analyze('exponential', 'T1_seed')

            T2 = popt[2]
            T2_err = np.sqrt(pcov[2][2])

            self.T2 = T2
            self.T2_err = T2_err

            t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1])

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec*1e6, self.data[1], 'o')
            if self.fit:
                ax.plot(t_vec_fit*1e6, exponential(t_vec_fit, *popt))
            ax.set_xlabel('Time (us)')
            ax.set_ylabel('Excited state probability')
            ax.set_title(self.element+f', n = {self.numpi}, T2CPMG_ =' + str(np.round(T2*1e6,4)) + '+/-' + str(np.round(T2_err*1e6,4)) + ' us')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)


    #
    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data['data'])
