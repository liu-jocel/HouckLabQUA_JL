from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import cosine
from LabCode2.Helpers.gate_macros import CZ

class CZ_total_phase(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,element_1,element_2,coupler,target,spectator,phases,num_phi,shots,calib_params,outer_folder='',path='CZ', suffix='data', cfg=None, plot=True, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, cfg=cfg)

        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.target = target
        self.spectator = spectator
        self.coupler = coupler
        self.phi_min = -0.5
        self.phi_max = 1.5
        self.num_phi = num_phi
        self.phi_vec = np.linspace(self.phi_min, self.phi_max, num_phi)  # Raw phi array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion
        self.phases = phases

        self.acquire()
        self.analyze()


    def make_prog_nopi(self):

        qubit_1 = self.element_1['q_name']
        qubit_2 = self.element_2['q_name']
        coupler = self.coupler['c_name']
        target = self.target['q_name']
        spectator = self.spectator['q_name']
        reset_ticks = self.element_1['reset_ticks']
        phi_vec = self.phi_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            p = declare(int)
            phis = declare(fixed,value=phi_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(p, 0, p < len(phi_vec), p + 1):
                    phi = phis[p]
                    wait(reset_ticks, qubit_1)
                    align()
                    play('X90', target)
                    align()
                    CZ(self.element_1,self.element_2,self.coupler,self.phases[0]+phi,self.phases[1]+phi)
                    align()
                    # frame_rotation_2pi(phi, target)
                    # frame_rotation_2pi(self.phases[1], spectator)
                    # frame_rotation_2pi(phi,qubit_1)
                    play('Xm90', target)

                    I, Q = measure_1q(self.target,  I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def make_prog_pi(self):

        qubit_1 = self.element_1['q_name']
        qubit_2 = self.element_2['q_name']
        coupler = self.coupler['c_name']
        target = self.target['q_name']
        spectator = self.spectator['q_name']
        reset_ticks = self.element_1['reset_ticks']
        phi_vec = self.phi_vec
        # amp_vec = self.amp_vec/self.coupler_amp_norm #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            p = declare(int)
            phis = declare(fixed,value=phi_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(p, 0, p < len(phi_vec), p + 1):
                    phi = phis[p]
                    wait(reset_ticks, qubit_1)
                    align()
                    play('X90', target)
                    play('X180', spectator)
                    align()
                    CZ(self.element_1, self.element_2, self.coupler, self.phases[0] + phi, self.phases[1] + phi)
                    align()
                    play('Xm90', target)

                    I, Q = measure_1q(self.target,  I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def acquire(self):

        self.job_nopi = super().acquire(self.make_prog_nopi())
        ss_array_no_pi = super().job_to_shots_1q(self.job_nopi, [self.shots, self.num_phi], self.calib_params)
        self.data = [self.phi_vec, np.mean(ss_array_no_pi, 0)]
        popt_nopi, pcov_nopi = super().analyze('cosine', 'phase_seed')

        if popt_nopi[0] < 0:
            popt_nopi[0] = popt_nopi[0] * -1
            popt_nopi[2] = popt_nopi[2] + 0.5

        self.job_pi = super().acquire(self.make_prog_pi())
        ss_array_pi = super().job_to_shots_1q(self.job_pi, [self.shots, self.num_phi], self.calib_params)
        self.data = [self.phi_vec, np.mean(ss_array_pi, 0)]
        popt_pi, pcov_pi = super().analyze('cosine', 'phase_seed')

        if popt_pi[0] < 0:
            popt_pi[0] = popt_pi[0] * -1
            popt_pi[2] = popt_pi[2] + 0.5
        self.full_data = [self.phi_vec, np.mean(ss_array_no_pi, 0), np.mean(ss_array_pi, 0)]
        self.fit_params = [popt_nopi,popt_pi]


    def simulate(self, channels, ticks, ticks_to_plot):

        self.samples = super().simulate(self.make_prog_pi(), channels, ticks, ticks_to_plot)

    def analyze(self):

        popt_nopi = self.fit_params[0]
        popt_pi = self.fit_params[1]

        self.phase = (popt_nopi[2] - popt_pi[2])%1

        self.correction = popt_pi[2]

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.phi_vec, self.full_data[1], 'o', label='no pi')
            ax.plot(self.phi_vec, cosine(self.phi_vec, *popt_nopi), label='no pi fit')
            ax.plot(self.phi_vec, self.full_data[2], 'o', label='pi')
            ax.plot(self.phi_vec, cosine(self.phi_vec, *popt_pi), label='pi fit')
            ax.axvline(x=self.correction)
            # ax.plot(self.phi_vec*1e9, counts_array[1]/self.shots, 'o', label='1')
            ax.set_xlabel('Phase applied (rad)')
            ax.set_ylabel('State probability')
            ax.set_title(f'CZ Total Phase Tuning, total $\phi$ = {self.phase}')
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])