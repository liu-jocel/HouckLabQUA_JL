from qm.qua import *
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


def _fft_peaks(t_vec, signal, min_peaks=3, rel_cutoff=0.5, min_sep_bins=5):
    """Return frequencies of dominant FFT peaks (units: 1/t_vec units).

    Takes all peaks above rel_cutoff * max_peak_amplitude, deduplicates peaks
    within min_sep_bins frequency bins of each other, with a minimum of
    min_peaks returned regardless of the cutoff.
    """
    dt = t_vec[1] - t_vec[0]
    freqs = np.fft.rfftfreq(len(t_vec), d=dt)
    fft_mag = np.abs(np.fft.rfft(signal - np.mean(signal)))
    mask = freqs > 0
    pos_freqs = freqs[mask]
    pos_mag = fft_mag[mask]
    min_sep = min_sep_bins * (pos_freqs[1] - pos_freqs[0])

    def _dedup(indices):
        kept = []
        for idx in indices:
            if all(abs(pos_freqs[idx] - pos_freqs[k]) >= min_sep for k in kept):
                kept.append(idx)
        return np.array(kept, dtype=int)

    peaks, _ = find_peaks(pos_mag)
    if len(peaks) == 0:
        top = np.argsort(pos_mag)[::-1][:max(min_peaks, 1)]
        return pos_freqs[top]

    peak_mags = pos_mag[peaks]
    qualified = peaks[peak_mags >= rel_cutoff * peak_mags.max()]
    by_mag = _dedup(qualified[np.argsort(pos_mag[qualified])[::-1]])

    if len(by_mag) < min_peaks:
        all_sorted = peaks[np.argsort(peak_mags)[::-1]]
        for idx in all_sorted:
            if len(by_mag) >= min_peaks:
                break
            if all(abs(pos_freqs[idx] - pos_freqs[k]) >= min_sep for k in by_mag):
                by_mag = np.append(by_mag, idx)

    return pos_freqs[by_mag]


def _make_ramsey_model(n_freqs):
    """Return a function: A * exp(-t/T2) * sum_i sin(2π f_i t + φ_i) + offset."""
    def model(t, A, T2, offset, *args):
        total = np.zeros_like(t, dtype=float)
        for i in range(n_freqs):
            total += np.sin(2 * np.pi * args[2 * i] * t + args[2 * i + 1])
        return A * np.exp(-t / T2) * total + offset
    return model


def _ramsey_p0_bounds(t_vec, signal, peak_freqs):
    """Initial guess and bounds for _make_ramsey_model with given peak_freqs."""
    A = (np.max(signal) - np.min(signal)) / 2
    offset = float(np.mean(signal[-10:]))
    T2 = (t_vec[-1] - t_vec[0]) / 3
    p0 = [A, T2, offset]
    lo = [-np.inf, 1e-3, -np.inf]
    hi = [np.inf, np.inf, np.inf]
    for f in peak_freqs:
        p0 += [float(f), np.pi / 2]
        lo += [0.0, -2 * np.pi]
        hi += [np.inf, 2 * np.pi]
    return p0, (lo, hi)

class T2R(ExperimentClass):
    """
    T2R experiment
    """
    def __init__(self, element, t_vec, shots, detuning, calib_params, outer_folder='', path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element
        self.t_vec = t_vec
        self.shots = shots
        self.detuning = detuning
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec,
            'shots': self.shots,
            'detuning': self.detuning,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        detuned_freq = self.meta_dict['IF_freq'] + self.detuning
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    update_frequency(qubit, detuned_freq)
                    play('X90', qubit)
                    wait(t, qubit)
                    play('X90', qubit)
                    update_frequency(qubit, self.meta_dict['IF_freq'])

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).save_all("ss")

        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")
        t_vec = self.t_vec / u.us
        dt = t_vec[1] - t_vec[0]

        matplotlib.use("qtagg", force=True)
        fig, (ax, ax_fft) = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 8))

        # Ramsey subplot
        line, = ax.plot([], [], 'o', ms=3, label='data')
        fit_line, = ax.plot([], [], '-', lw=1.5, alpha=0.8, label='fit')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Excited state probability')
        ax.set_xlim(-10, t_vec[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # FFT subplot
        pos_freqs_khz = np.fft.rfftfreq(len(t_vec), d=dt)[1:] * 1000  # MHz → kHz
        fft_line, = ax_fft.plot(pos_freqs_khz, np.zeros(len(pos_freqs_khz)), lw=1.5)
        peak_markers = ax_fft.scatter([], [], color='red', zorder=5, s=40, label='peaks')
        ax_fft.set_xlabel('Frequency (kHz)')
        ax_fft.set_ylabel('FFT magnitude')
        ax_fft.set_xlim(0, pos_freqs_khz[-1])
        ax_fft.legend()

        fig.suptitle(f"{self.element}, T2R")
        interrupt_on_close(fig, job)

        ss_all = None
        while results.is_processing():
            ss_all = results.fetch_all()[0]
            if ss_all.shape[0] == 0:
                continue
            ss_array = np.mean(ss_all, axis=0)
            print(f"Shot: {ss_all.shape[0]}/{self.shots}", end="\r", flush=True)
            line.set_data(t_vec, ss_array)

            # Update FFT subplot
            fft_mag = np.abs(np.fft.rfft(ss_array - np.mean(ss_array)))[1:]
            fft_line.set_ydata(fft_mag)
            try:
                peak_freqs = _fft_peaks(t_vec, ss_array)
                peak_khz = peak_freqs * 1000
                peak_mags = fft_mag[[np.argmin(np.abs(pos_freqs_khz - f)) for f in peak_khz]]
                peak_markers.set_offsets(np.c_[peak_khz, peak_mags])
            except Exception:
                pass
            ax_fft.relim()
            ax_fft.autoscale_view(scalex=False)

            if ss_all.shape[0] > 100:
                try:
                    peak_freqs = _fft_peaks(t_vec, ss_array)
                    model = _make_ramsey_model(len(peak_freqs))
                    p0, bounds = _ramsey_p0_bounds(t_vec, ss_array, peak_freqs)
                    popt, pcov = curve_fit(model, t_vec, ss_array, p0=p0, bounds=bounds, maxfev=10000)
                    T2, T2_err = popt[1], np.sqrt(pcov[1][1])
                    freqs_fit = [popt[3 + 2*i] for i in range(len(peak_freqs))]
                    t_fit = np.linspace(t_vec[0], t_vec[-1], 400)
                    fit_line.set_data(t_fit, model(t_fit, *popt))
                    freq_str = ", ".join(f"{f*1000:.1f}" for f in freqs_fit)
                    ax.set_title(f"T2R = {T2:.4f} ± {T2_err:.4f} µs  freqs=[{freq_str}] kHz")
                except Exception as e:
                    print(f"\nLive fit: {e}", end="")
            ax.relim()
            ax.autoscale_view(scalex=False)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        print("")
        ss_array = np.mean(ss_all, axis=0) if ss_all is not None else np.zeros(len(t_vec))
        self.data.update({'ss_data': ss_array})

        try:
            peak_freqs = _fft_peaks(t_vec, ss_array)
            model = _make_ramsey_model(len(peak_freqs))
            p0, bounds = _ramsey_p0_bounds(t_vec, ss_array, peak_freqs)
            popt, pcov = curve_fit(model, t_vec, ss_array, p0=p0, bounds=bounds, maxfev=10000)
            T2 = popt[1]
            T2_err = np.sqrt(pcov[1][1])
            freqs_fit = [popt[3 + 2*i] for i in range(len(peak_freqs))]
            self.T2 = T2
            self.T2_err = T2_err
            self.data.update({'T2R': (T2, T2_err), 'fit_popt': popt, 'fit_freqs': freqs_fit})
            freq_str = ", ".join(f"{f*1000:.3f}" for f in freqs_fit)
            ax.set_title(f"{self.element} T2R = {T2:.4f} ± {T2_err:.4f} us\nfreqs = [{freq_str}] kHz")
            print(f"T2R = {T2:.4f} ± {T2_err:.4f} us,  freqs = [{freq_str}] kHz")
        except Exception as e:
            print(f"T2R Fit Failed: {e}")
            ax.set_title(f"{self.element}, T2R")

        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        plt.ioff()
