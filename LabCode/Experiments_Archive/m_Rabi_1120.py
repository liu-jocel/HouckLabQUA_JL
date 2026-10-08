from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import decaying_cosine
from LabCode2.Helpers.gate_macros import flat_top_CZ

class Rabi_1120(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self,element_1,element_2,coupler,t_min,t_max,num_t,shots,calib_params,outer_folder='',path='CZ', suffix='data', meta_dict=None, plot=True, fit=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element_1 = element_1 #meta dict element name
        self.element_2 = element_2
        self.coupler = coupler
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        self.t_vec_raw = np.linspace(t_min, t_max, num_t)  # Raw time array made from input parameters
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw)  # Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec * 4e-9  # Convert to physical time array
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit

        self.acquire()
        self.analyze()

    def make_prog(self):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        coupler = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        clk_vec = self.clk_vec #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            c = declare(int)
            clks = declare(int,value=clk_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                # for a in amp_vec:
                with for_(c, 0, c < len(clk_vec), c + 1):
                    clk = clks[c]
                    wait(reset_ticks, qubit_1)
                    align()
                    play('X180', qubit_1)
                    play('X180', qubit_2)
                    align()
                    flat_top_CZ(self.coupler, clk)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.element_1], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def acquire(self):

        self.job = super().acquire(self.make_prog())

    def simulate(self, channels, ticks, ticks_to_plot):

        self.samples = super().simulate(self.make_prog(), channels, ticks, ticks_to_plot)

    def analyze(self):

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_t], self.calib_params)
        self.data = [self.t_vec, np.mean(ss_array, 0)]

        if self.fit:
            popt, pcov = super().analyze('decaying_cosine', 'rabi_seed')

            if popt[0] < 0:
                popt[0] = popt[0] * -1
                popt[2] = popt[2] + 0.5

            self.pulse_time = int(1/popt[1]*0.25e9)*4
        else:
            self.pulse_time = 0

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec, self.data[1], 'o')
            if self.fit:
                ax.plot(self.t_vec, decaying_cosine(self.t_vec, *popt), label='fit')
            ax.axvline(x=self.pulse_time*1e-9)
            ax.set_title(f'11 to 20 Rabi, pulse time = {self.pulse_time*1e-9}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Excited state probability')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])