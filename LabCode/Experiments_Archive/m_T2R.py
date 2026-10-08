from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import decaying_cosine, cosine

class T2R(ExperimentClass):
    """
    T2R experiement
    """
    def __init__(self,element,t_min,t_max,num_t,shots, detuning, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, plot=True,freq_cal=False, fit=True, save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element
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
        if freq_cal:
            self.analyze_freq_cal()
        else:
            self.analyze()

        if self.save:
            self.pickle_data()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        clk_vec = self.clk_vec
        shots = self.shots
        detuned_freq = self.meta_dict[self.element]['q_IF'] + self.detuning

        with program() as prog:
            n = declare(int)
            c = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            clks = declare(int, value=clk_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            update_frequency(qubit, detuned_freq)

            with for_(n, 0, n < shots, n + 1):
                with for_(c, 0, c< len(clk_vec), c+1):
                    clk = clks[c]
                    wait(reset_ticks, qubit)
                    align()
                    play('X90', qubit)
                    wait(clk, qubit)
                    align()
                    play('X90', qubit)
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
            popt, pcov = super().analyze('decaying_cosine', 'T2R_seed', detuning = self.detuning)

            T2 = popt[4]
            T2_err = np.sqrt(pcov[4][4])

            self.T2 = T2
            self.T2_err = T2_err

            self.freq_correction = popt[1] - self.detuning
            self.freq_err = pcov[1][1]

            t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1], 200)

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec*1e6, self.data[1], 'o')
            if self.fit:
                ax.plot(t_vec_fit*1e6, decaying_cosine(t_vec_fit, *popt))
            ax.set_xlabel('Time (us)')
            ax.set_ylabel('Excited state probability')
            ax.set_title(self.element+', T2R = ' + str(np.round(T2*1e6,4)) + '+/-' + str(np.round(T2_err*1e6,4)) + ' us')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def analyze_freq_cal(self):

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params)
        self.data = [self.t_vec, np.mean(ss_array, 0)]

        popt, pcov = super().analyze('decaying_cosine', 'T2R_seed', detuning = self.detuning)

        self.freq_correction = popt[1] - self.detuning
        self.freq_err = np.sqrt(pcov[1][1])

        t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1])

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, self.data[1], 'o')
            ax.plot(t_vec_fit, decaying_cosine(t_vec_fit, *popt))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            ax.set_title(self.element+', Correction = ' + str(np.round(self.freq_correction/1e6,4)) + ' MHz')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    #
    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data['data'])