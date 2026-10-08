from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode2.Helpers.fit_functions import decaying_cosine
from LabCode2.Helpers.gate_macros import flat_top_CZ
from LabCode2.Helpers.SPAM_correction import correct_counts_array

class coupler_temp_duty(ExperimentClass):
    """
    Swap oscillation experiement
    """
    def __init__(self,element_1,element_2,coupler,m_min,m_max,num_m,wait_ticks,shots,calib_params,outer_folder='',path='temp', suffix='data', meta_dict=None, plot=True, fit=False, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.coupler = coupler
        self.coupler_amp_norm = meta_dict[coupler]['pulse_amp']
        self.m_min = m_min
        self.m_max = m_max
        self.num_m = num_m
        self.m_vec = np.linspace(m_min,m_max,num_m,dtype=int)
        self.wait_ticks = wait_ticks
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.confusion = confusion

        self.job = self.acquire(self.make_prog())
        self.analyze()


    def make_prog(self):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        coupler = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        m_vec = self.m_vec
        shots = self.shots
        wait_ticks = self.wait_ticks

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            m = declare(int)
            i = declare(int)
            ms = declare(int, value=m_vec)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(m, 0, m < self.num_m, m + 1):
                    num_gates = ms[m]
                    wait(reset_ticks, qubit_1)
                    align()
                    with for_(i, 0, i<num_gates, i+1):
                        play('iSWAP_gauss', coupler)
                        wait(wait_ticks, coupler)
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                    save(I1, I1_stream)
                    save(Q1, Q1_stream)
                    save(I2, I2_stream)
                    save(Q2, Q2_stream)

            with stream_processing():
                I1_stream.save_all("I_c1")
                Q1_stream.save_all("Q_c1")
                I2_stream.save_all("I_c2")
                Q2_stream.save_all("Q_c2")

        return prog

    def analyze(self):

        counts_array = super().job_to_shots_2q(self.job, [self.shots, self.num_m], self.calib_params)
        counts_array = correct_counts_array(self.confusion,counts_array)
        self.data = [self.m_vec, counts_array]

        res_handles = self.job.result_handles

        I_1 = res_handles.get("I_c1").fetch_all()['value']
        Q_1 = res_handles.get("Q_c1").fetch_all()['value']
        I_2 = res_handles.get("I_c2").fetch_all()['value']
        Q_2 = res_handles.get("Q_c2").fetch_all()['value']

        if self.fit:
            popt, pcov = super().analyze('decaying_cosine', 'rabi_seed',ydata=counts_array[1]/self.shots)

            if popt[0] < 0:
                popt[0] = popt[0] * -1
                popt[2] = popt[2] + 0.5

            self.pulse_time = int(1/popt[1]*0.125e9)*4
        else:
            self.pulse_time = 0

        if self.plot:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
            # ax.plot(self.t_vec*1e9, counts_array[0]/self.shots, 'o', label='00')
            # ax.plot(self.t_vec*1e9, counts_array[1]/self.shots, 'o', label='01')
            # ax.plot(self.t_vec*1e9, counts_array[2]/self.shots, 'o', label='10')
            # ax.plot(self.t_vec*1e9, counts_array[3]/self.shots, 'o', label='11')
            ax[0].plot(self.m_vec, (counts_array[0] + counts_array[2]) / self.shots, 'o', label='Q1 0')
            ax[0].plot(self.m_vec, (counts_array[1] + counts_array[3]) / self.shots, 'o', label='Q1 1')
            ax[0].plot(self.m_vec, (counts_array[0] + counts_array[1]) / self.shots, 'o', label='Q2 0')
            ax[0].plot(self.m_vec, (counts_array[2] + counts_array[3]) / self.shots, 'o', label='Q2 1')
            ax[1].plot(I_1,Q_1,'.')
            ax[1].set_xlim(-1e-4, 2e-4)
            ax[1].set_ylim(-1e-4, 1e-4)
            ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
            ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
            if self.fit:
                ax.plot(self.t_vec*1e9, decaying_cosine(self.t_vec, *popt), label='fit')
            ax[0].set_xlabel('Number of cycles')
            ax[0].set_ylabel('State probability')
            ax[0].set_title('Temperature calibration data')
            ax[0].legend()
            plt.savefig(self.iname[:-4]+"LO_"+str(self.meta_dict['c1']['c_LO']['LO_freq']/1e6)+
                                             "_amp_"+str(self.meta_dict['c1']['pulse_amp'])
                                             +"_duty.png", bbox_inches="tight")
            plt.close(fig)


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])