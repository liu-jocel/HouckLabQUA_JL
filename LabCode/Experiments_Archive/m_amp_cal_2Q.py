from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import measure_1q

class AmpCal2Q(ExperimentClass):
    """
    CZ amplitude calibation
    """
    def __init__(self,qubit_1,qubit_2,coupler,err_range,num_errs,num_pis,shots,calib_params,outer_folder='',path='amp_cal', suffix='data', cfg=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, cfg=cfg)

        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        self.coupler = coupler
        self.err_range = err_range
        self.num_errs = num_errs
        self.num_pis = num_pis
        self.err_vec = np.linspace(1-self.err_range,1+self.err_range,num_errs)
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot

        self.acquire()
        self.analyze()

    def make_prog(self):

        coupler = self.coupler['c_name']
        qubit_1 = self.qubit_1['q_name']
        qubit_2 = self.qubit_2['q_name']
        reset_ticks = self.qubit_1['reset_ticks']
        err_vec = self.err_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            e = declare(int)
            p = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            err_vec = declare(fixed, value=err_vec)
            I_stream_even = declare_stream()
            Q_stream_even = declare_stream()
            I_stream_odd = declare_stream()
            Q_stream_odd = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(e, 0, e < self.num_errs, e+1):
                    wait(reset_ticks, qubit_1)
                    align()
                    play('X180', qubit_1)
                    play('X180', qubit_2)
                    align()
                    play('iSWAP_const'*amp(0.25), coupler)
                    with for_(p, 0, p < self.num_pis, p+1):
                        play('iSWAP_const'*amp(err_vec[e]), coupler)
                    align()
                    I, Q = measure_1q(self.qubit_1, I, Q)

                    save(I, I_stream_even)
                    save(Q, Q_stream_even)

                    align()
                    wait(reset_ticks, qubit_1)
                    align()
                    play('X180', qubit_1)
                    play('X180', qubit_2)
                    align()
                    play('iSWAP_const' * amp(0.25*err_vec[e]), coupler)
                    with for_(p, 0, p < self.num_pis, p + 1):
                        play('iSWAP_const' * amp(err_vec[e]), coupler)
                    play('iSWAP_const' * amp(0.5*err_vec[e]), coupler)
                    align()
                    I, Q = measure_1q(self.qubit_1, I, Q)

                    save(I, I_stream_odd)
                    save(Q, Q_stream_odd)

            with stream_processing():
                I_stream_even.save_all("I_even")
                Q_stream_even.save_all("Q_even")
                I_stream_odd.save_all("I_odd")
                Q_stream_odd.save_all("Q_odd")

        return prog

    def acquire(self):

        self.job = super().acquire(self.make_prog())

    def simulate(self, channels, ticks, ticks_to_plot):

        self.samples = super().simulate(self.make_prog(), channels, ticks, ticks_to_plot)

    def analyze(self):

        ss_array_even = super().job_to_shots_1q(self.job, [self.shots, self.num_errs], self.calib_params, I_name='I_even', Q_name='Q_even')
        ss_array_odd = super().job_to_shots_1q(self.job, [self.shots, self.num_errs], self.calib_params, I_name='I_odd', Q_name='Q_odd')
        self.data = [self.err_vec, np.mean(ss_array_even, 0), np.mean(ss_array_odd,0)]

        self.power_factor = self.err_vec[np.argmin(np.abs(self.data[1]-self.data[2]))]

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.err_vec, self.data[1]-self.data[2], 'o')
            # ax.plot(self.err_vec, self.data[2], 'o')
            ax.set_xlabel('Amplitude factor')
            ax.set_ylabel('Abs. Even-odd difference')
            ax.set_title('CZ drive, power factor = ' + str(self.power_factor))
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])