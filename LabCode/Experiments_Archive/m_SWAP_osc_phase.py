from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from LabCode2.Helpers.gate_macros import flat_top_iSWAP_Z_corr,flat_top_iSWAP_dg_Z_corr, update_frame
from LabCode2.Helpers.fit_functions import iSWAP_phase

class SWAP_Osc_phase(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,target,spectator,coupler,num_phi,phi_corr,shots,calib_params,outer_folder='',path='SWAP', suffix='data', meta_dict=None, plot=True, fit=True, do_acquire=True, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.target = target
        self.spectator = spectator
        self.coupler = coupler
        self.phi_min = -1
        self.phi_max = 1
        self.num_phi = num_phi
        self.phi_vec = np.linspace(self.phi_min, self.phi_max, num_phi)  # Raw phi array made from input parameters
        self.phi_step = self.phi_vec[1]-self.phi_vec[0]
        self.beta_vec = 2*np.pi*self.phi_vec/2
        self.phi_corr = phi_corr
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion
        self.fit = fit
        self.do_acquire = do_acquire

        if self.do_acquire:
            self.job = self.acquire(self.make_prog())
            self.analyze()


    def make_prog(self):


        target = self.target
        spectator = self.spectator
        coupler = self.coupler
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        phi_vec = self.phi_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):

                for p in self.phi_vec:
                    reset_frame()
                    wait(reset_ticks, target)
                    align()
                    play('X90', target)
                    play('X180', spectator)
                    align()
                    flat_top_iSWAP_Z_corr('q1','q2',coupler,self.phi_corr,self.phi_corr)
                    # frame_rotation_2pi(self.phi_corr, target)
                    flat_top_iSWAP_dg_Z_corr('q1','q2',coupler,self.phi_corr,self.phi_corr)
                    # frame_rotation_2pi(self.phi_corr, target)
                    align()
                    frame_rotation_2pi(p, target)
                    play('X90', target)

                    I, Q = measure_1q(self.meta_dict[self.target], I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog


    def analyze(self):

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_phi], self.calib_params)
        # counts_array = np.linalg.inv(self.confusion)@counts_array
        self.data = [self.beta_vec, np.mean(ss_array,0)]

        if self.fit:
            popt, pcov = super().analyze('iSWAP_phase', 'iSWAP_phase_seed')

            phi_corr = popt[1]
            phi_corr_err = np.sqrt(pcov[1][1])

            if popt[0]>0:

                self.phi_corr = -phi_corr/(2*np.pi)
                self.phi_corr_err = phi_corr_err/(2*np.pi)

            else:

                self.phi_corr = (-phi_corr+np.pi/2)/(2*np.pi)
                self.phi_corr_err = phi_corr_err/(2*np.pi)

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.beta_vec, self.data[1], 'o', label='0')
            if self.fit:
                ax.plot(self.beta_vec, iSWAP_phase(self.beta_vec,*popt),label='fit')
                ax.vlines(self.phi_corr*2*np.pi,0,1,'k')
            # ax.plot(self.phi_vec*1e9, counts_array[1]/self.shots, 'o', label='1')
            ax.set_xlabel('Phase applied (rad)')
            ax.set_ylabel('State probability')
            ax.set_title(f'phi_corr = {self.phi_corr:.3f} +/- {self.phi_corr_err:.3f}')
            ax.legend()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])