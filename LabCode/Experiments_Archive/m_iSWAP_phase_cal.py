from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import iSWAP_phase
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg

class iSWAP_PhaseCal(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,target,spectator,coupler,num_beta,shots,calib_params,outer_folder='',path='SWAP', suffix='data', meta_dict=None, plot=True, run=True, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.target = target
        self.spectator = spectator
        self.coupler = coupler
        self.beta_min = -0.2
        self.beta_max = 0.5
        self.num_beta = num_beta
        self.beta_vec = np.linspace(self.beta_min, self.beta_max, num_beta)  # Raw beta array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion

        if run:
            self.acquire()
            self.analyze()
            self.save_data()
        else:
            self.meta_dict['global_params']['reset_ticks'] = 4


    def make_prog_I(self):

        target = self.target
        spectator = self.spectator
        coupler = self.coupler
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        beta_vec = self.beta_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            p = declare(int)
            betas = declare(fixed,value=beta_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(p, 0, p < len(beta_vec), p + 1):
                    beta = betas[p]
                    reset_frame(target,spectator,coupler)
                    # wait_for_trigger(target)
                    wait(reset_ticks, target)
                    align()
                    # play('X90', target)
                    # align()
                    # flat_top_iSWAP(coupler)
                    # # frame_rotation_2pi(0, target)
                    # # frame_rotation_2pi(0, spectator)
                    # # frame_rotation_2pi(0, coupler)
                    # align()
                    # flat_top_iSWAP_dg(coupler)
                    # # frame_rotation_2pi(0, target)
                    # # frame_rotation_2pi(0, spectator)
                    # # frame_rotation_2pi(0, coupler)
                    # align()
                    # frame_rotation_2pi(-2*beta, target)
                    # play('X90', target)
                    # align()
                    I, Q = measure_1q(self.meta_dict[self.target],  I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def make_prog_pi2(self):

        target = self.target
        spectator = self.spectator
        coupler = self.coupler
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        beta_vec = self.beta_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            p = declare(int)
            betas = declare(fixed, value=beta_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(p, 0, p < len(beta_vec), p + 1):
                    beta = betas[p]
                    reset_frame(target,spectator,coupler)
                    # wait_for_trigger(target)
                    wait(reset_ticks, target)
                    align()
                    play('X90', target)
                    # play('X90', spectator)
                    # align()
                    # flat_top_iSWAP(coupler)
                    # align()
                    # flat_top_iSWAP_dg(coupler)
                    # align()
                    # frame_rotation_2pi(-2 * beta, target)
                    # play('X90', target)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.target], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def make_prog_pi(self):

        target = self.target
        spectator = self.spectator
        coupler = self.coupler
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        beta_vec = self.beta_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            p = declare(int)
            betas = declare(fixed, value=beta_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(p, 0, p < len(beta_vec), p + 1):
                    beta = betas[p]
                    reset_frame(target,spectator,coupler)
                    # wait_for_trigger(target)
                    wait(reset_ticks, target)
                    align()
                    play('X90', target)
                    # play('X180', spectator)
                    # align()
                    # flat_top_iSWAP(coupler)
                    # align()
                    # flat_top_iSWAP_dg(coupler)
                    # align()
                    # frame_rotation_2pi(-2 * beta, target)
                    play('X90', target)
                    align()
                    I, Q = measure_1q(self.meta_dict[self.target], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def acquire(self):

        self.job_I = super().acquire(self.make_prog_I())
        ss_array_I = super().job_to_shots_1q(self.job_I, [self.shots, self.num_beta], self.calib_params)
        self.data = [self.beta_vec, np.mean(ss_array_I, 0)]
        popt_I, pcov_I = super().analyze('iSWAP_phase', 'iSWAP_phase_seed')

        if popt_I[0] < 0:
            popt_I[0] = popt_I[0] * -1
            popt_I[1] = popt_I[1] + 0.25

        self.job_pi2 = super().acquire(self.make_prog_pi2())
        ss_array_pi2 = super().job_to_shots_1q(self.job_pi2, [self.shots, self.num_beta], self.calib_params)
        self.data = [self.beta_vec, np.mean(ss_array_pi2, 0)]
        popt_pi2, pcov_pi2 = super().analyze('iSWAP_phase', 'iSWAP_phase_seed')

        if popt_pi2[0] < 0:
            popt_pi2[0] = popt_pi2[0] * -1
            popt_pi2[1] = popt_pi2[1] + 0.25

        self.job_pi = super().acquire(self.make_prog_pi())
        ss_array_pi = super().job_to_shots_1q(self.job_pi, [self.shots, self.num_beta], self.calib_params)
        self.data = [self.beta_vec, np.mean(ss_array_pi, 0)]
        popt_pi, pcov_pi = super().analyze('iSWAP_phase', 'iSWAP_phase_seed')

        if popt_pi[0] < 0:
            popt_pi[0] = popt_pi[0] * -1
            popt_pi[1] = popt_pi[1] + 0.25
        self.full_data = [self.beta_vec, np.mean(ss_array_I, 0), np.mean(ss_array_pi2, 0), np.mean(ss_array_pi, 0)]
        self.fit_params = [popt_I,popt_pi2,popt_pi]

    def analyze(self):

        popt_I = self.fit_params[0]
        popt_pi2 = self.fit_params[1]
        popt_pi = self.fit_params[2]

        self.correction = -popt_pi2[1]

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.beta_vec, self.full_data[1], 'o', label='I')
            ax.plot(self.beta_vec, iSWAP_phase(self.beta_vec, *popt_I), label='I fit')
            ax.plot(self.beta_vec, self.full_data[2], 'o', label='pi2')
            ax.plot(self.beta_vec, iSWAP_phase(self.beta_vec, *popt_pi2), label='pi2 fit')
            ax.plot(self.beta_vec, self.full_data[3], 'o', label='pi')
            ax.plot(self.beta_vec, iSWAP_phase(self.beta_vec, *popt_pi), label='pi fit')
            ax.axvline(x=self.correction)
            # ax.plot(self.phi_vec*1e9, counts_array[1]/self.shots, 'o', label='1')
            ax.set_xlabel('Phase applied (rad)')
            ax.set_ylabel('State probability')
            # ax.set_title(f'CZ Total Phase Tuning, total $\phi$ = {self.phase}')
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])