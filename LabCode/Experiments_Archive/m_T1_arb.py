from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q, measure_3q
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.gate_macros import flat_top_CZ, flat_top_iSWAP
from LabCode2.Helpers.SPAM_correction import correct_counts_array


class T1_arb(ExperimentClass):
    """
    Swap oscillation experiment
    """
    def __init__(self,element, coupler,t_min,t_max,num_t,shots,discrim_class,outer_folder='',path='SWAP', suffix='data', meta_dict=None, plot=True, fit=False, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.discrim = discrim_class
        self.element = element
        self.coupler = coupler
        self.coupler_amp_norm = meta_dict[coupler]['pulse_amp']
        self.t_min = t_min
        self.t_max = t_max
        self.num_t = num_t
        self.t_vec_raw = np.linspace(t_min, t_max, num_t)  # Raw time array made from input parameters np.logspace(np.log10(t_min),np.log10(t_max),num_t) #
        self.clk_vec = time_vec_to_clk_vec_wait(self.t_vec_raw)  # Convert to a clock tick array consistent with the wait() command in qua
        self.t_vec = self.clk_vec * 4e-9  # Convert to physical time array
        # self.amp_vec = np.linspace(self.a_min, self.a_max, self.num_a) #Raw amplitude array made from input parameters
        self.shots = shots
        # self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.confusion = confusion

        self.job = self.acquire(self.make_prog())
        self.analyze()


    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        coupler = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        clk_vec = self.clk_vec
        # amp_vec = self.amp_vec/self.coupler_amp_norm #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            Ic = declare(fixed)
            Qc = declare(fixed)
            c = declare(int)
            clks = declare(int,value=clk_vec)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()
            Ic_stream = declare_stream()
            Qc_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(c, 0, c < len(clk_vec), c + 1):
                    clk = clks[c]
                    wait_for_trigger(qubit)
                    wait(reset_ticks, qubit)
                    align()
                    play('X180', qubit)
                    align()
                    # # play('iSWAP_gauss', self.coupler, duration=clk)
                    flat_top_CZ(self.coupler, clk)#  if commented out, put it back
                    # wait(clk, qubit) # if not, comment it out
                    align()
                    I1, Q1, I2, Q2, Ic, Qc = measure_3q(self.meta_dict["q1"], self.meta_dict["q2"],
                                                        self.meta_dict["qc"],
                                                        I1, Q1, I2, Q2, Ic, Qc)

                    save(I1, I1_stream)
                    save(Q1, Q1_stream)
                    save(I2, I2_stream)
                    save(Q2, Q2_stream)
                    save(Ic, Ic_stream)  #
                    save(Qc, Qc_stream)  #

            with stream_processing():
                I1_stream.save_all("I1")
                Q1_stream.save_all("Q1")
                I2_stream.save_all("I2")
                Q2_stream.save_all("Q2")
                Ic_stream.save_all("Ic")
                Qc_stream.save_all("Qc")

        return prog

    def analyze(self):
        counts_dict = super().job_to_shots_arb(self.job, [self.shots, self.num_t], self.discrim)
        # counts_array = np.linalg.inv(self.confusion)@counts_array
        counts_array = np.zeros((len(counts_dict), self.num_t))
        for j, key in enumerate(self.discrim.POVM.keys()):
            counts_array[j, :] = counts_dict[key]

        counts_array = correct_counts_array(self.confusion, counts_array)

        self.data = [self.t_vec, counts_array, list(self.discrim.POVM.keys()),self.confusion]

        if self.fit:
            popt, pcov = super().analyze('exponential', 'T1_seed', ydata=counts_array[1])
            T1 = popt[2]
            T1_err = np.sqrt(pcov[2][2])

            self.T1 = T1
            self.T1_err = T1_err

            t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1], 100)

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)

            for c,state in enumerate(self.discrim.POVM.keys()):
                ax.plot(self.t_vec * 1e9, counts_array[c], 'o', label=state)

            if self.fit:
                pass
                ax.plot(t_vec_fit* 1e9, exponential(t_vec_fit, *popt))
                ax.set_title(self.element + ', T1 = ' + str(np.round(T1 * 1e6, 4)) + '+/-' + str(
                np.round(T1_err * 1e6, 4)) + ' us')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('State probability')
            ax.legend()
            # plt.semilogx()
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)