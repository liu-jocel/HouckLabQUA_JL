from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiChevronIQ(ExperimentClass):
    """
    Rabi chevron experiment reading out IQ data
    """
    def __init__(self, element, amp_vec, f_vec, shots, num_pi_pulses=1, pulse_type="X180", outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

        self.element = element
        self.f_vec = f_vec
        self.pulse_type = pulse_type
        self.amp_vec_abs = amp_vec
        self.amp_vec = self.amp_vec_abs / meta_dict[pulse_type]['amp']
        self.shots = shots
        self.num_pi = num_pi_pulses

        self.read_len = self.meta_dict['global_params']['read_len']

        self.data = {
            'qubit': self.element,
            "pulse_type": self.pulse_type,
            "number_pulses": self.num_pi,
            "shots": self.shots,
            "f_vec": self.f_vec,
            "amp_vec": self.amp_vec,
            "meta_dict": self.meta_dict
        }

        self.config = config
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        IF_freq = self.meta_dict["IF_freq"]
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
                    with for_each_(df, self.f_vec):
                        update_frequency(qubit, IF_freq + df)
                        with for_(*from_array(a, self.amp_vec)):
                            wait(reset_time * u.ns, qubit)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                play('X180' * amp(a), qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).save_all("I")
                    Q_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).save_all("Q")

        if self.pulse_type: # == "X90":
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
                    with for_each_(df, self.f_vec):
                        update_frequency(qubit, IF_freq + df)
                        with for_(*from_array(a, self.amp_vec)):
                            wait(reset_time * u.ns, qubit)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                # play('X90' * amp(a), qubit)
                                # play('X90' * amp(a), qubit)
                                play(self.pulse_type * amp(a), qubit)
                                play(self.pulse_type * amp(a), qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            save(I, I_stream)
                            save(Q, Q_stream)
        # elif self.pulse_type == "Y90":
        #     with program() as prog:
        #         n = declare(int)
        #         I = declare(fixed)
        #         Q = declare(fixed)
        #         df = declare(int)
        #         a = declare(fixed)
        #         m = declare(int)
        #         I_stream = declare_stream()
        #         Q_stream = declare_stream()
        #         with for_(n, 0, n < shots, n + 1):
        #             with for_each_(df, self.f_vec):
        #                 update_frequency(qubit, IF_freq + df)
        #                 with for_(*from_array(a, self.amp_vec)):
        #                     wait(reset_time * u.ns, qubit)
        #                     with for_(m, 0, m < self.num_pi, m + 1):
        #                         play("X90" * amp(a), qubit)
        #                         play("X90" * amp(a), qubit)
        #                         align(qubit, resonator)
        #                     I, Q = measure_1q(self.meta_dict, I, Q)
        #                     save(I, I_stream)
        #                     save(Q, Q_stream)
                with stream_processing():
                    I_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).save_all("I")
                    Q_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).save_all("Q")
        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["I", "Q"], mode="live")

        dummy = np.zeros((len(self.f_vec), len(self.amp_vec_abs)))
        fig, (ax_I, ax_Q) = plt.subplots(1, 2, figsize=(10, 4))
        im_I = ax_I.pcolormesh(self.amp_vec_abs, self.f_vec / u.MHz, dummy, shading='nearest')
        ax_I.set_xlabel("Qubit pulse amplitude [V]")
        ax_I.set_ylabel("Qubit detuning [MHz]")
        fig.colorbar(im_I, ax=ax_I)
        ax_I.set_title("I Data")
        im_Q = ax_Q.pcolormesh(self.amp_vec_abs, self.f_vec / u.MHz, dummy, shading='nearest')
        ax_Q.set_xlabel("Qubit pulse amplitude [V]")
        ax_Q.set_ylabel("Qubit detuning [MHz]")
        fig.colorbar(im_Q, ax=ax_Q)
        ax_Q.set_title("Q Data")
        fig.suptitle(f"({self.element}, {self.pulse_type}) {(self.meta_dict['q_LO']['LO_freq'] + self.meta_dict['IF_freq']) / u.MHz:.3f} MHz. {self.num_pi} pulses")
        plt.tight_layout()
        interrupt_on_close(fig, job)

        star_I = ax_I.scatter([], [], marker='*', s=300, c='red', zorder=5)
        star_Q = ax_Q.scatter([], [], marker='*', s=300, c='red', zorder=5)

        I_all = None
        Q_all = None
        I_array = dummy.copy()
        Q_array = dummy.copy()
        while results.is_processing():
            I_all, Q_all = results.fetch_all()
            if I_all.dtype.names:
                I_all = I_all['value']
                Q_all = Q_all['value']
            if I_all.shape[0] == 0:
                continue
            I_array = np.mean(I_all, axis=0)
            Q_array = np.mean(Q_all, axis=0)
            im_I.set_array(I_array.ravel())
            im_I.set_clim(I_array.min(), I_array.max())
            im_Q.set_array(Q_array.ravel())
            im_Q.set_clim(Q_array.min(), Q_array.max())
            c_live = I_array + 1j * Q_array
            dist_live = np.abs(c_live - c_live[:, 0:1])
            smoothed_live = gaussian_filter(dist_live, sigma=2)
            freq_idx_live = int(np.argmin(np.abs(self.f_vec)))
            row_live = smoothed_live[freq_idx_live]
            peaks_live, _ = find_peaks(row_live)
            amp_idx_live = int(peaks_live[0]) if len(peaks_live) > 0 else int(np.argmax(row_live))
            _opt_amp_live = self.amp_vec_abs[amp_idx_live]
            _opt_det_live = self.f_vec[freq_idx_live] / u.MHz
            star_I.set_offsets([[_opt_amp_live, _opt_det_live]])
            star_Q.set_offsets([[_opt_amp_live, _opt_det_live]])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)
        c = I_array + 1j * Q_array
        dist = np.abs(c - c[:, 0:1])
        smoothed = gaussian_filter(dist, sigma=2)
        freq_idx = int(np.argmin(np.abs(self.f_vec)))
        row = smoothed[freq_idx]
        # Find peaks with both prominence and height requirements to filter out noise
        row_max = np.max(row)
        row_range = row_max - np.min(row)
        min_prominence = row_range * 0.25  # At least 25% of signal range prominence
        min_height = row_max * 0.5  # At least 50% of maximum height
        peaks, _ = find_peaks(row, prominence=min_prominence, height=min_height)
        if len(peaks) > 0:
            amp_idx = int(peaks[0])
        else:
            # Fallback: if no peaks meet both criteria, try relaxed prominence
            peaks, _ = find_peaks(row, prominence=row_range * 0.1)
            if len(peaks) > 0:
                amp_idx = int(peaks[0])
            else:
                # Final fallback: take the maximum
                amp_idx = int(np.argmax(row))
        opt_amp = self.amp_vec_abs[amp_idx]
        opt_detuning_Hz = int(self.f_vec[freq_idx])
        print(f"First π-pulse at {opt_detuning_Hz / u.MHz:.3f} MHz, {opt_amp:.4f} V")
        print(job.execution_report())

        self.data.update({
            'I': I_array,
            'Q': Q_array,
            'raw_IQ': np.stack([I_all, Q_all]) if I_all is not None else None,
            'opt_amp': opt_amp,
            'opt_detuning_Hz': opt_detuning_Hz,
        })

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])
