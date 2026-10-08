from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('C:/Users/my/Desktop/Houck-Lab-Qua')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)



class ChiMeasurement(ExperimentClass):
    """
    Transmission experiement
    """
    def __init__(self,element,f_min,f_max,f_step,shots,outer_folder='',path='trans', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.f_min = f_min
        self.f_max = f_max
        self.f_step = f_step
        # assert f_step >= 0.1e6, 'Frequency step must be at least 0.1 MHz'
        self.f_vec = np.arange(f_min, f_max, f_step)
        self.num_f = len(self.f_vec)
        self.shots = shots
        self.meta_dict = meta_dict
        self.config = config
        
        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            "meta_dict": self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), config=self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        f_vec = self.f_vec
        shots = self.shots
        IF_freq = self.meta_dict['IF_freq']
        anharmonicity = self.meta_dict['anharmonicity']

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            I_stream_ground = declare_stream()
            Q_stream_ground = declare_stream()
            I_stream_excited = declare_stream()
            Q_stream_excited = declare_stream()
            I_stream_f = declare_stream()
            Q_stream_f = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(f, 0, f < len(f_vec), f + 1):
                    freq = f_vec[0]+self.f_step*f
                    update_frequency(resonator, freq)

                    wait(reset_time * u.ns, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_ground)
                    save(Q, Q_stream_ground)

                    align()

                    wait(reset_time * u.ns, resonator)
                    align()

                    play('X180', self.element)
                    align()
                    update_frequency(self.element, IF_freq + anharmonicity)
                    play('X180_ef', self.element)
                    update_frequency(self.element, IF_freq)

                    align()

                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_f)
                    save(Q, Q_stream_f)

                    align()

                    wait(reset_time * u.ns, resonator)

                    align()

                    play('X180', self.element)


                    align()

                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_excited)
                    save(Q, Q_stream_excited)

                    align()

            with stream_processing():
                I_stream_ground.save_all("I_ground")
                Q_stream_ground.save_all("Q_ground")
                I_stream_excited.save_all("I_excited")
                Q_stream_excited.save_all("Q_excited")
                I_stream_f.save_all("I_f")
                Q_stream_f.save_all("Q_f")

        return prog

    def analyze(self):

        IQ_array_g = super().job_to_IQ_avg_1q(self.job, [self.shots, self.num_f],
                                              stream_tag_I='I_ground', stream_tag_Q='Q_ground')
        IQ_array_e = super().job_to_IQ_avg_1q(self.job, [self.shots, self.num_f],
                                              stream_tag_I='I_excited', stream_tag_Q='Q_excited')
        IQ_array_f = super().job_to_IQ_avg_1q(self.job, [self.shots, self.num_f],
                                              stream_tag_I='I_f', stream_tag_Q='Q_f')

        # --- Electrical delay correction on shot-averaged trajectories ---
        I_g_raw = np.mean(IQ_array_g[0], axis=0)
        Q_g_raw = np.mean(IQ_array_g[1], axis=0)
        I_e_raw = np.mean(IQ_array_e[0], axis=0)
        Q_e_raw = np.mean(IQ_array_e[1], axis=0)
        I_f_raw = np.mean(IQ_array_f[0], axis=0)
        Q_f_raw = np.mean(IQ_array_f[1], axis=0)

        # Fit delay from ground state; apply same correction to all three
        _, _, tau = self._correct_electrical_delay(I_g_raw, Q_g_raw, self.f_vec)
        correction = np.exp(-1j * 2 * np.pi * tau * self.f_vec)

        S21_g = (I_g_raw + 1j * Q_g_raw) * correction
        S21_e = (I_e_raw + 1j * Q_e_raw) * correction
        S21_f = (I_f_raw + 1j * Q_f_raw) * correction

        # Magnitude and phase from delay-corrected data
        log_mag_g = 20 * np.log10(np.abs(S21_g))
        log_mag_e = 20 * np.log10(np.abs(S21_e))
        log_mag_f = 20 * np.log10(np.abs(S21_f))

        phase_g = np.angle(S21_g)
        phase_e = np.angle(S21_e)
        phase_f = np.angle(S21_f)

        chi = (self.f_vec[np.argmin(log_mag_e)] - self.f_vec[np.argmin(log_mag_g)]) * 1e-6

        self.data.update({
            "log_mag_g": log_mag_g, "log_mag_e": log_mag_e, "log_mag_f": log_mag_f,
            "phase_g": phase_g, "phase_e": phase_e, "phase_f": phase_f,
            "tau_ns": tau * 1e9,
        })

        read_amp = self.meta_dict['read_amp']
        suptitle = (f"{self.meta_dict['r_name']}  |  "
                    f"χ = {chi:.4f} MHz  |  read amp: {read_amp}  |  τ = {tau * 1e9:.3f} ns")

        # --- Figure 1: transmission + phase ---
        fig1, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 6))
        f_MHz = self.f_vec * 1e-6

        for data, label, ls in zip(
                [log_mag_g, log_mag_e, log_mag_f],
                ['ground', 'excited', 'f state'],
                ['-', '--', ':']
        ):
            ax[0].plot(f_MHz, data, ls=ls, label=label)
        ax[0].set_xlabel('IF frequency (MHz)')
        ax[0].set_ylabel('Transmission (dB)')
        ax[0].legend(fontsize=8)

        for data, label, ls in zip(
                [phase_g, phase_e, phase_f],
                ['ground', 'excited', 'f state'],
                ['-', '--', ':']
        ):
            ax[1].plot(f_MHz, data, ls=ls, label=label)
        ax[1].set_xlabel('IF frequency (MHz)')
        ax[1].set_ylabel('Phase (rad)')
        ax[1].legend(fontsize=8)

        fig1.suptitle(suptitle, fontsize=9)
        plt.savefig(self.iname, bbox_inches='tight')
        #plt.close(fig1)

        # --- Figure 2: IQ circle plot ---
        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))
        self._plot_iq_circles(ax2, S21_g, S21_e, S21_f, title=self.meta_dict['r_name'], tau=tau)
        fig2.suptitle(suptitle, fontsize=9)
        circle_iname = self.iname.replace('.png', '_circle.png')
        #plt.savefig(circle_iname, bbox_inches='tight')
        #plt.close(fig2)

    def _correct_electrical_delay(self, I, Q, f_vec):
        S21 = I + 1j * Q
        n = len(f_vec)
        n_edge = max(3, n // 10)
        idx = np.r_[np.arange(n_edge), np.arange(n - n_edge, n)]
        phase = np.unwrap(np.angle(S21[idx]))
        coeffs = np.polyfit(f_vec[idx], phase, 1)
        tau = coeffs[0] / (2 * np.pi)
        correction = np.exp(-1j * 2 * np.pi * tau * f_vec)
        S21_corr = S21 * correction
        return S21_corr.real, S21_corr.imag, tau

    def _plot_iq_circles(self, ax, S21_g, S21_e, S21_f, title='', tau=0.0):
        """
        S21_g/e/f are already delay-corrected complex arrays of shape (num_f,).
        """
        ax.plot(S21_g.real, S21_g.imag, color='tab:green', lw=1.5, label='ground')
        ax.plot(S21_e.real, S21_e.imag, color='tab:red', lw=1.5, ls='--', label='excited')
        ax.plot(S21_f.real, S21_f.imag, color='tab:orange', lw=1.5, ls=':', label='f state')

        # Mark transmission minimum for each state
        for S21, color in [(S21_g, 'tab:green'), (S21_e, 'tab:red'), (S21_f, 'tab:orange')]:
            idx_min = np.argmin(np.abs(S21))
            ax.scatter(S21.real[idx_min], S21.imag[idx_min], zorder=5, color=color, s=40)

        ax.set_aspect('equal')
        ax.set_xlabel('I (a.u.)')
        ax.set_ylabel('Q (a.u.)')
        ax.set_title(f'{title}  |  τ = {tau * 1e9:.3f} ns')
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.axvline(0, color='gray', lw=0.4, ls='--')



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])

