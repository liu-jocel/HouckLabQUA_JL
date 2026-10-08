from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import measure_1q

class PowerFunnel(ExperimentClass):
    """
    Single qubit power funnel
    """
    def __init__(self,element,err_range,num_errs,num_pis,shots,calib_params,outer_folder='',path='amp_cal', suffix='data', meta_dict=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element
        self.err_range = err_range
        self.num_errs = num_errs
        self.num_pis = num_pis
        self.err_vec = np.linspace(1-self.err_range,1+self.err_range,num_errs)
        self.pi_vec = np.linspace(1,self.num_pis,self.num_pis,dtype=int)
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot

        self.job = self.acquire(self.make_prog())
        self.analyze()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        err_vec = self.err_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            e = declare(int)
            q = declare(int) # iterator to play pi pulses
            p = declare(int) # number of 2pi pulses
            I = declare(fixed)
            Q = declare(fixed)
            err_vec = declare(fixed, value=err_vec)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(e, 0, e < self.num_errs, e+1):
                    wait(reset_ticks, qubit)
                    with for_(p, 0, p < self.num_pis, p+1):
                        with for_(q, 0, q < p, q+1):
                            play('X180' * amp(err_vec[e]), qubit)
                            play('X180' * amp(err_vec[e]), qubit)
                        align()
                        I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                        save(I, I_stream)
                        save(Q, Q_stream)


            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog

    def analyze(self):

        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_errs, self.num_pis], self.calib_params, I_name='I', Q_name='Q')
        self.data = [self.err_vec, np.mean(ss_array, 0)]

        power_data = ss_array.mean(axis=(0,2))
        self.power_factor = self.err_vec[np.argmin(power_data)]

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            plot = ax.pcolormesh(self.pi_vec, self.err_vec, self.data[1])
            ax.set_xlabel('Number of 2$\pi$ pulses')
            ax.set_ylabel('Power factor')
            ax.set_title(self.element+', power factor = ' + str(self.power_factor))
            plt.colorbar(plot)

            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])