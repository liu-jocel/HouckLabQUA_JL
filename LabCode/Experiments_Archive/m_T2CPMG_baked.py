from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import exponential_2
from qualang_tools.loops import from_array
from LabCode2.Config.config_helpers import meta_dict_to_config_6Q
from qm import simulate
from LabCode2.Config.config_helpers import Omega_G, Omega_G_dot
import copy
from qualang_tools.bakery.bakery import baking

"""
CURRENTLY NOT WORKING 
"""
class T2CPMG_baked(ExperimentClass):
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

    def bake_sequence(self, t, n):
        element = self.element
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        self.baked_config = copy.deepcopy(meta_dict_to_config_6Q(self.meta_dict))

        pulse_amp = self.meta_dict[element]['pi_amp']
        pulse_len = self.meta_dict["global_params"]['gauss_len']
        DRAG_beta = self.meta_dict[element]['DRAG_beta']

        I = Omega_G(pulse_amp, pulse_len, pulse_len / 4)
        Q = Omega_G_dot(DRAG_beta * pulse_amp, pulse_len, pulse_len / 4)

        tau = t/n
        with baking(self.baked_config) as b:
            b.add_op('baked_x', element, [I, Q])
            b.add_op('baked_y', element, [Q, I])
            b.reset_frame(element)
            b.wait(reset_ticks)
            b.play('baked_x', element, amp=1/2)
            b.wait(int(tau/2), element)
            for _ in range(n-1):
                b.play('baked_y', element)
                b.wait(int(tau), element)
            b.play('baked_y', element)
            b.wait(int(tau/2), element)
            b.play('baked_x', element, amp=1/2)
        return b

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

                for t in clk_tvec:
                # with for_(*from_array(t, clk_tvec)):
                    b = self.bake_sequence(t,numpi)
                    b.run()
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
            popt, pcov = super().analyze('exponential', 'T2E_seed')

            T2 = popt[2]
            T2_err = np.sqrt(pcov[2][2])

            self.T2 = T2
            self.T2_err = T2_err

            t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1])

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.t_vec*1e6, self.data[1], 'o')
            if self.fit:
                ax.plot(t_vec_fit*1e6, exponential_2(t_vec_fit, *popt))
            ax.set_xlabel('Time (us)')
            ax.set_ylabel('Excited state probability')
            ax.set_title(self.element+', T2CPMG = ' + str(np.round(T2*1e6,4)) + '+/-' + str(np.round(T2_err*1e6,4)) + ' us')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)


    #
    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data['data'])
