from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.fit_functions import cosine
from LabCode2.Helpers.gate_macros import CZ

class CZ_phase(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,element_1,element_2,target,spectator,coupler,num_phi,shots,calib_params,outer_folder='',path='CZ', suffix='data', cfg=None, plot=True, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, cfg=cfg)

        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.target = target
        self.spectator = spectator
        self.coupler = coupler
        self.coupler_amp_norm = coupler['pulse_amp']
        self.phi_min = -0.5
        self.phi_max = 1.5
        self.num_phi = num_phi
        self.phi_vec = np.linspace(self.phi_min, self.phi_max, num_phi)  # Raw phi array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion

        self.acquire()
        self.analyze()


    def make_prog(self):

        qubit_1 = self.element_1['q_name']
        qubit_2 = self.element_2['q_name']
        target = self.target['q_name']
        spectator = self.spectator['q_name']
        coupler = self.coupler['c_name']
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
                    wait(reset_ticks, target)
                    play('X90', target)
                    align()
                    CZ(self.element_1,self.element_2,self.coupler,phi,phi)
                    align()
                    # frame_rotation_2pi(phi, target)
                    play('X90', target)

                    I, Q = measure_1q(self.target,  I, Q)

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

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_phi], self.calib_params)
        # counts_array = np.linalg.inv(self.confusion)@counts_array
        self.data = [self.phi_vec, np.mean(ss_array,0)]

        popt, pcov = super().analyze('cosine', 'phase_seed')

        if popt[0] < 0:
            popt[0] = popt[0]*-1
            popt[2] = popt[2] + 0.5

        self.correction = popt[2]

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.phi_vec, self.data[1], 'o', label='0')
            ax.plot(self.phi_vec, cosine(self.phi_vec,*popt), label = 'fit')
            ax.axvline(x=popt[2])
            ax.set_xlabel('Phase applied (rad)')
            ax.set_ylabel('State probability')
            ax.set_title(f'CZ Local Phase Tuning, correction = {popt[2]}')
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])