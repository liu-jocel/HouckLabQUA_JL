from qm.qua import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib

matplotlib.use("qtagg", force=True)
from datetime import datetime
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import cosine, T2R_seed
from qualang_tools.results import progress_counter, fetching_tool
from LabCode.Helpers.pulse_macros import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class ZZCrosstalk(ExperimentClass):
    """
    T2R experiement on element 1 when element 2 is in 0 or 1
    """
    def __init__(self,element1, element2, t_vec, shots, detuning, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element1 = element1 #meta dict element
        self.element2 = element2
        self.t_vec = t_vec
        self.shots = shots
        self.detuning = detuning
        self.calib_params = calib_params
        self.data = {
            "qubit1": self.element1,
            "qubit2": self.element2,
            "t_vec": self.t_vec,
            'shots': self.shots,
            'detuning': self.detuning,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit1 = self.meta_dict[self.element1][self.element1]['q_name']
        qubit2 = self.meta_dict[self.element2][self.element2]['q_name']
        resonator = self.meta_dict[self.element1][self.element1]['r_name']
        detuned_freq = self.meta_dict[self.element1][self.element1]['IF_freq'] + self.detuning
        reset_time = self.meta_dict[self.element1][self.element1]['reset_time']
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
            ss_stream_0 = declare_stream()
            ss_stream_1 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):

                    ### Prepare qubit 2 in 0
                    update_frequency(qubit1, detuned_freq)
                    play_pulse(qubit1, "X90",
                               phase=self.meta_dict[self.element1][self.element1]["X90"]['phase'])
                    wait(t, qubit1)
                    play_pulse(qubit1, "X90",
                               phase=self.meta_dict[self.element1][self.element1]["X90"]['phase'])

                    # SS Readout
                    align(qubit1, qubit2, resonator)
                    I, Q = measure_1q(self.meta_dict[self.element1][self.element1], I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)

                    save(state, ss_stream_0)
                    align(qubit1, qubit2, resonator)
                    wait(reset_time*u.ns, qubit1)

                    ### Prepare qubit 2 in 1
                    align(qubit1, qubit2, resonator)
                    play("X90", qubit2)
                    play("X90", qubit2)
                    align(qubit1, qubit2, resonator)

                    update_frequency(qubit1, detuned_freq)
                    play_pulse(qubit1, "X90",
                               phase=self.meta_dict[self.element1][self.element1]["X90"]['phase'])
                    wait(t, qubit1)
                    play_pulse(qubit1, "X90",
                               phase=self.meta_dict[self.element1][self.element1]["X90"]['phase'])

                    # SS Readout
                    align(qubit1, qubit2, resonator)
                    I, Q = measure_1q(self.meta_dict[self.element1][self.element1], I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)

                    save(state, ss_stream_1)
                    align(qubit1, qubit2, resonator)
                    wait(reset_time * u.ns, qubit1)
            with stream_processing():
                ss_stream_0.buffer(len(self.t_vec)).average().save("ss_0")
                ss_stream_1.buffer(len(self.t_vec)).average().save("ss_1")
        return prog

    def analyze(self, job, qm):
        # Extract data
        result = fetching_tool(self.job, data_list=["ss_0", "ss_1"], mode='live')

        # Live Plotting
        fig, ax = plt.subplots()
        fig.suptitle(f"{self.element1} ZZ Crosstalk Measurement with {self.element2}")
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Excited state probability')
        # --- Initialize lines before loop ---
        (line0,) = ax.plot(self.t_vec, [0.5] * len(self.t_vec), "o", label="q2=0")
        (line1,) = ax.plot(self.t_vec, [0.5] * len(self.t_vec), "o", label="q2=1")

        (fit0,) = ax.plot(self.t_vec, [0.5] * len(self.t_vec), "-", lw=1)
        (fit1,) = ax.plot(self.t_vec, [0.5] * len(self.t_vec), "-", lw=1)

        ax.legend()

        # --- Helper fit function ---
        def fit_ramsey(t, y):
            amplitude_guess = (np.max(y) - np.min(y)) / 2
            offset_guess = np.mean(y[-10:])
            phase_guess = 90

            fft_freqs = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
            fft_signal = np.fft.fft(y - offset_guess)

            pos = fft_freqs > 0
            dominant_freq = fft_freqs[pos][np.argmax(np.abs(fft_signal[pos]))]

            p0 = [amplitude_guess, dominant_freq, phase_guess, offset_guess]
            popt, pcov = curve_fit(cosine, t, y, p0=p0, maxfev=5000)
            return popt, pcov

        # --- Live loop ---
        while result.is_processing():
            pop_0, pop_1 = result.fetch_all()

            self.data.update({
                'pop_0': pop_0,
                'pop_1': pop_1
            })

            # --- Update data points ---
            line0.set_data(self.t_vec, pop_0)
            line1.set_data(self.t_vec, pop_1)

            # --- Fit both traces ---
            try:
                popt0, pcov0 = fit_ramsey(self.t_vec, pop_0)
                popt1, pcov1 = fit_ramsey(self.t_vec, pop_1)

                t_fit = np.linspace(self.t_vec[0], self.t_vec[-1], 200)

                fit0.set_data(t_fit, cosine(t_fit, *popt0))
                fit1.set_data(t_fit, cosine(t_fit, *popt1))

                # --- Extract ZZ shift ---
                f0 = popt0[1] * u.GHz
                f1 = popt1[1] * u.GHz
                ZZ = (f1 - f0) / u.MHz

                ax.set_title(f"ZZ shift: {ZZ:.3f} MHz")

                self.data['fit_0'] = {'popt': popt0, 'pcov': pcov0}
                self.data['fit_1'] = {'popt': popt1, 'pcov': pcov1}
                self.data['ZZ_MHz'] = ZZ

            except Exception:
                pass

            # --- Refresh plot ---
            ax.relim()
            ax.autoscale_view(scaley=False)
            plt.pause(0.05)

            plt.pause(0.01)
        plt.savefig(self.iname, bbox_inches="tight")
