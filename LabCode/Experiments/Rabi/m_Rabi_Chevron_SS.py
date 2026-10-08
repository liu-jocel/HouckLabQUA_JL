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
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiChevronSS(ExperimentClass):
    """
    Rabi chevron experiment with single shot readout
    """
    def __init__(self, element, amp_vec, f_vec, shots, calib_params, num_pi_pulses=1, pulse_type="X180", outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

        self.element = element
        self.f_vec = f_vec
        self.pulse_type = pulse_type
        self.amp_vec_abs = amp_vec
        self.amp_vec = self.amp_vec_abs / meta_dict[pulse_type]['amp']
        self.shots = shots
        self.num_pi = num_pi_pulses
        self.calib_params = calib_params

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
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        IF_freq = self.meta_dict['IF_freq']

        if self.pulse_type == "X180":
            with program() as prog:
                n = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                df = declare(int)
                a = declare(fixed)
                m = declare(int)
                state = declare(int)
                I_reset = declare(fixed)
                ss_stream = declare_stream()

                with for_(n, 0, n < shots, n + 1):
                    with for_each_(df, self.f_vec):
                        with for_each_(a, self.amp_vec):
                            update_frequency(qubit, IF_freq + df)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                play('X180' * amp(a), qubit)
                                align(qubit, resonator)

                            # SS Readout
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            with if_(I > ss_thresh):
                                assign(state, 1)
                            with else_():
                                assign(state, 0)
                            save(state, ss_stream)
                            align(qubit, resonator)

                            # Active reset
                            update_frequency(qubit, IF_freq)
                            with while_(I_reset > ground_confidence_threshold):
                                with if_(I_reset > ss_thresh):
                                    play('X90', qubit)
                                    play('X90', qubit)
                                    align(qubit, resonator)
                                I, Q = measure_1q(self.meta_dict, I, Q)
                                assign(I_reset, I)
                                wait(resonator_dump_time * u.ns, qubit)

                with stream_processing():
                    ss_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).average().save("ss")

        if self.pulse_type : # handle both X90 and Y90  == "X90":
            with program() as prog:
                n = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                df = declare(int)
                a = declare(fixed)
                m = declare(int)
                state = declare(int)
                I_reset = declare(fixed)
                ss_stream = declare_stream()

                with for_(n, 0, n < shots, n + 1):
                    with for_each_(df, self.f_vec):
                        with for_each_(a, self.amp_vec):
                            update_frequency(qubit, IF_freq + df)
                            with for_(m, 0, m < self.num_pi, m + 1):
                                # play('X90' * amp(a), qubit)
                                # play('X90' * amp(a), qubit)
                                play(self.pulse_type * amp(a), qubit)
                                play(self.pulse_type * amp(a), qubit)
                                align(qubit, resonator)

                            # SS Readout
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            with if_(I > ss_thresh):
                                assign(state, 1)
                            with else_():
                                assign(state, 0)
                            save(state, ss_stream)
                            align(qubit, resonator)

                            # Active reset
                            update_frequency(qubit, IF_freq)
                            with while_(I_reset > ground_confidence_threshold):
                                with if_(I_reset > ss_thresh):
                                    play('X180', qubit)
                                    align(qubit, resonator)
                                I, Q = measure_1q(self.meta_dict, I, Q)
                                assign(I_reset, I)
                                wait(resonator_dump_time * u.ns, qubit)

                with stream_processing():
                    ss_stream.buffer(len(self.amp_vec)).buffer(len(self.f_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")

        dummy = np.zeros((len(self.f_vec), len(self.amp_vec_abs)))
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.pcolormesh(self.amp_vec_abs, self.f_vec / u.MHz, dummy, shading='nearest', vmin=0, vmax=1)
        ax.set_xlabel("Qubit pulse amplitude [V]")
        ax.set_ylabel("Qubit detuning [MHz]")
        fig.colorbar(im, ax=ax, label='P(excited)')
        fig.suptitle(f"({self.element}, {self.pulse_type}) {(self.meta_dict['q_LO']['LO_freq'] + self.meta_dict['IF_freq']) / u.MHz:.3f} MHz. {self.num_pi} pulses")
        plt.tight_layout()
        interrupt_on_close(fig, job)

        star = ax.scatter([], [], marker='*', s=300, c='red', zorder=5)

        ss_array = dummy.copy()
        while results.is_processing():
            ss_array = results.fetch_all()[0]
            if ss_array.shape != (len(self.f_vec), len(self.amp_vec_abs)):
                continue
            im.set_array(ss_array.ravel())
            im.set_clim(ss_array.min(), ss_array.max())
            smoothed_live = gaussian_filter(ss_array, sigma=2)

            # Find first peak at each frequency and maximize contrast
            first_peaks_amp_live = []
            first_peaks_contrast_live = []
            for freq_idx in range(len(self.f_vec)):
                amplitude_slice = smoothed_live[freq_idx, :]
                peaks, properties = find_peaks(amplitude_slice, height=0.1)
                if len(peaks) > 0:
                    first_peak_idx = peaks[0]
                    first_peaks_amp_live.append(self.amp_vec_abs[first_peak_idx])
                    first_peaks_contrast_live.append(properties['peak_heights'][0])
                else:
                    max_idx = np.argmax(amplitude_slice)
                    first_peaks_amp_live.append(self.amp_vec_abs[max_idx])
                    first_peaks_contrast_live.append(amplitude_slice[max_idx])

            best_freq_idx_live = np.argmax(first_peaks_contrast_live)
            opt_amp_live = first_peaks_amp_live[best_freq_idx_live]
            star.set_offsets([[opt_amp_live, self.f_vec[best_freq_idx_live] / u.MHz]])
            ax.set_title(f"π-pulse: {self.f_vec[best_freq_idx_live] / u.MHz:.3f} MHz, {opt_amp_live:.4f} V")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        smoothed = gaussian_filter(ss_array, sigma=2)

        # Detect Rabi oscillations: find first peak at each frequency and maximize contrast
        first_peaks_amp = []
        first_peaks_contrast = []
        for freq_idx in range(len(self.f_vec)):
            amplitude_slice = smoothed[freq_idx, :]
            peaks, properties = find_peaks(amplitude_slice, height=0.1)  # Find peaks with min height 0.1
            if len(peaks) > 0:
                first_peak_idx = peaks[0]  # Take first (smallest amplitude) peak
                first_peaks_amp.append(self.amp_vec_abs[first_peak_idx])
                first_peaks_contrast.append(properties['peak_heights'][0])  # Peak contrast/height
            else:
                # If no peaks found, use the max (fallback)
                max_idx = np.argmax(amplitude_slice)
                first_peaks_amp.append(self.amp_vec_abs[max_idx])
                first_peaks_contrast.append(amplitude_slice[max_idx])

        # Find frequency with best-contrast first peak
        best_freq_idx = np.argmax(first_peaks_contrast)
        opt_amp = first_peaks_amp[best_freq_idx]
        opt_detuning_Hz = int(self.f_vec[best_freq_idx])
        print(f"First π-pulse at {opt_detuning_Hz / u.MHz:.3f} MHz, {opt_amp:.4f} V")

        self.data.update({"ss_data": ss_array,
                          "opt_amp": opt_amp,
                          "opt_detuning_Hz": opt_detuning_Hz})
        print(job.execution_report())
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])