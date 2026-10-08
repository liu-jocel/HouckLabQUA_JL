from qm.qua import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_1q
from qualang_tools.loops import from_array
from LabCode2.Helpers.fit_functions import decaying_cosine
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
from qualang_tools.results.data_handler import DataHandler


class Rabi_test(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self,element, amp_vec, f_vec, shots, calib_params, num_pi_pulses=1, pulse_type= "X180", outer_folder='', path='Rabi', suffix='data', meta_dict=None, ss=False,plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element name
        self.calib_params = calib_params
        self.f_vec = f_vec
        self.pulse_type = pulse_type
        self.X180_amp_norm = self.meta_dict[self.element]['pi_amp']
        self.X90_amp_norm = self.meta_dict[self.element]['pi2_amp']
        self.amp_vec_abs = amp_vec
        if self.pulse_type == 'X180':
            self.amp_vec = self.amp_vec_abs/self.X180_amp_norm
        if self.pulse_type == 'X90':
            self.amp_vec = self.amp_vec_abs/self.X90_amp_norm

        self.shots = shots
        self.num_pi = num_pi_pulses
        self.ss = ss
        self.plot = plot

        self.read_len = self.meta_dict['global_params']['read_len']
        self.job = self.acquire(self.make_prog())

        self.data = {
            "shots": self.shots,
            "f_vec": self.f_vec,
            "amp_vec": self.amp_vec,
            "meta_dict": self.meta_dict
        }

        if ss:
            self.analyze_ss()
        else:
            self.analyze()

    def make_prog(self):

        qubit = self.meta_dict[self.element]['q_name']
        reset_ticks = self.meta_dict[self.element]['reset_ticks']
        shots = self.shots
        if self.pulse_type == "X180":
            with program() as prog:
                n = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                df = declare(int)
                a = declare(fixed)
                m = declare(int)
                I_stream = declare_stream()
                Q_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_(*from_array(df, self.f_vec)):
                        with for_(*from_array(a, self.amp_vec)):
                            wait(reset_ticks, qubit)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                play('X180' * amp(a), qubit)

                            I, Q = measure_1q(self.meta_dict[self.element], I, Q)
                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.save_all("I")
                    Q_stream.save_all("Q")

        if self.pulse_type == "X90":
            with program() as prog:
                n = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                df = declare(int)
                a = declare(fixed)
                m = declare(int)
                I_stream = declare_stream()
                Q_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_(*from_array(df, self.f_vec)):
                        with for_(*from_array(a, self.amp_vec)):
                            wait(reset_ticks, qubit)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                play('X90' * amp(a), qubit)
                                play('X90' * amp(a), qubit)

                            I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).average().save("I")
                    Q_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).average().save("Q")
        return prog

    def analyze(self):
        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, len(self.amp_vec), len(self.f_vec)])
        I_array = np.mean(IQ_array[0], 0)
        Q_array = np.mean(IQ_array[1], 0)
        self.data.update({
            'I': I_array,
            'Q': Q_array,
            })
        plt.suptitle(f"{self.element} ({(self.meta_dict[self.element]['q_LO']['LO_freq'] + self.meta_dict[self.element]['q_IF']) / u.MHz}[MHz])")
        plt.subplot(121)
        plt.cla()
        plt.pcolor(self.amp_vec_abs, self.f_vec / u.MHz, I_array)
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.colorbar()
        plt.title("I Data")
        plt.subplot(122)
        plt.cla()
        plt.pcolor(self.amp_vec_abs, self.f_vec / u.MHz, Q_array)
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.title("Q Data")
        plt.tight_layout()
        plt.colorbar()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()
        # fig = plt.figure()
        # interrupt_on_close(fig, self.job)
        # results = fetching_tool(self.job, ['I ', 'Q'], mode='live')
        # while results.is_processing():
        #     I, Q = results.fetch_all()
        #     I, Q = u.demod2volts(I, self.read_len), u.demod2volts(Q, self.read_len)
        #     # Plots
        #     plt.suptitle("Rabi chevron")
        #     plt.subplot(121)
        #     plt.cla()
        #     plt.pcolor(self.amp_vec, self.f_vec / u.MHz, I)
        #     plt.xlabel("Qubit pulse amplitude [V]")
        #     plt.ylabel("Qubit 1 detuning [MHz]")
        #     plt.title(f"q1 (f_res: {(self.meta_dict[self.element]['Q_LO']['LO_freq'] + self.meta_dict[self.element]['Q_IF']) / u.MHz} MHz)")
        #     plt.subplot(122)
        #     plt.cla()
        #     plt.pcolor(self.amp_vec, self.f_vec / u.MHz, Q)
        #     plt.xlabel("Qubit pulse amplitude [V]")
        #     plt.ylabel("Qubit 1 detuning [MHz]")
        #     plt.tight_layout()
        #     plt.pause(1)
        #
        #     self.data.update({
        #         'I': I,
        #         'Q': Q,
        #     })
        #     super().save_data(self.data)
        # plt.savefig(self.iname, bbox_inches="tight")
        # plt.close(fig)

    def analyze_ss(self):
        ss_array = super().job_to_shots_1q(self.job, [self.shots, self.num_a], self.calib_params)
        self.data = [self.amp_vec, np.mean(ss_array, 0)]

        # popt, pcov = super().analyze('decaying_cosine', 'rabi_seed')
        #
        # if popt[0] < 0:
        #     popt[0] = popt[0] * -1
        #     popt[2] = popt[2] + 0.5
        #
        # self.pulse_amp = 0.5/popt[1]

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.amp_vec*self.X180_amp_norm, self.data[1], 'o')
            # ax.plot(self.amp_vec, decaying_cosine(self.amp_vec,*popt))
            ax.set_xlabel('Amp (V)')
            ax.set_ylabel('Excited state probability')
            # ax.set_title(self.element+', Pi amp = ' + str(np.round(self.pulse_amp,4)))
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])