from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class QubitSpec(ExperimentClass):
    """
    Qubit spectroscopy experiement
    """
    def __init__(self, element, f_min, f_max, f_step, shots, outer_folder='', path='spec', suffix='data', meta_dict=None,
                 config=None, plot=True, save=True):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.f_min = f_min
        self.f_max = f_max
        self.f_step = f_step
        self.f_vec = np.arange(f_min, f_max, f_step)
        self.num_f = len(self.f_vec)
        self.shots = shots
        self.config = config
        self.plot = plot
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            'shots': self.shots,
            "meta_dict": self.meta_dict
        }

        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if save:
            self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        reset_time = self.meta_dict['reset_time']
        f_vec = self.f_vec
        shots = self.shots
        resonator = self.meta_dict['r_name']
        freqr = self.meta_dict['r_IF']
        resonator_dump_time = self.meta_dict['resonator_dump_time']

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):

                with for_each_(f, f_vec):
                    update_frequency(qubit, f)
                    play("constant_tone", qubit)
                    # play("X90", qubit)
                    align()
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream)
                    save(Q, Q_stream)
                    wait(resonator_dump_time*u.ns)

            with stream_processing():
                I_stream.buffer(self.num_f).save_all("I")
                Q_stream.buffer(self.num_f).save_all("Q")

        return prog

    def analyze(self, job, qm):
        def if_MHz_to_qubit_freq_MHz(if_freq_MHz):
            return np.array(self.meta_dict['q_LO']['LO_freq'] + if_freq_MHz * u.MHz).T / u.MHz

        def qubit_freq_MHz_to_if_MHz(qubit_freq_MHz):
            return np.array(qubit_freq_MHz * u.MHz - self.meta_dict['q_LO']['LO_freq'] * u.GHz).T / u.MHz

        results = fetching_tool(job, data_list=["I", "Q"], mode="live")

        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
        line_mag, = ax_mag.plot(self.f_vec / u.MHz, np.zeros(self.num_f))
        ax_mag.set_ylabel('Transmission (dB)')
        ax_mag.set_title(f'{self.element} Qubit Spec')
        ax2 = ax_mag.secondary_xaxis('top', functions=(if_MHz_to_qubit_freq_MHz, qubit_freq_MHz_to_if_MHz))
        ax2.set_xlabel("Qubit frequency (GHz)")
        line_phase, = ax_phase.plot(self.f_vec / u.MHz, np.zeros(self.num_f))
        ax_phase.set_xlabel('IF frequency (MHz)')
        ax_phase.set_ylabel('Phase (rad)')
        interrupt_on_close(fig, job)

        I_array = np.zeros(self.num_f)
        Q_array = np.zeros(self.num_f)
        while results.is_processing():
            I_all, Q_all = results.fetch_all()
            if I_all.shape[0] == 0:
                continue
            I_array = np.mean(I_all, axis=0)
            Q_array = np.mean(Q_all, axis=0)
            mag = 20 * np.log10(np.sqrt(I_array**2 + Q_array**2))
            phase = np.arctan2(Q_array, I_array)
            line_mag.set_ydata(mag)
            ax_mag.relim()
            ax_mag.autoscale_view()
            line_phase.set_ydata(phase)
            ax_phase.relim()
            ax_phase.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        self.data.update({
            "IQ_magnitude_dBm": mag,
            "IQ_phase": phase,
        })
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])