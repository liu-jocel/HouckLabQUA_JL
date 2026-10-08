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

class ACStarkRamsey(ExperimentClass):
    """
    T2R experiement
    """
    def __init__(self,element, t_vec, shots, ramsey_detuning, stark_detuning, stark_amp, calib_params,outer_folder='',path='T2R_Stark', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.t_vec = t_vec
        self.shots = shots
        self.ramsey_detuning = ramsey_detuning
        self.stark_detuning = stark_detuning
        self.stark_amp = stark_amp
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec,
            'shots': self.shots,
            'ramsey_detuning': self.ramsey_detuning,
            'stark_detuning': self.stark_detuning,
            'stark_amp': self.stark_amp,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        ramsey_detuned_freq = self.meta_dict['IF_freq'] + self.ramsey_detuning
        stark_detuned_freq = self.meta_dict['IF_freq'] + self.stark_detuning
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
                    update_frequency(qubit, ramsey_detuned_freq)
                    play_pulse(qubit, "X90",
                               phase=self.meta_dict["X90"]['phase'])
                    align(qubit, resonator)
                    update_frequency(qubit, stark_detuned_freq)
                    play("constant_tone", qubit, duration=t)
                    align(qubit, resonator)
                    update_frequency(qubit, ramsey_detuned_freq)
                    play_pulse(qubit, "X90",
                               phase=self.meta_dict["X90"]['phase'])

                    align(qubit, resonator)
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

                    update_frequency(qubit, self.meta_dict['IF_freq'])
                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play_pulse(qubit, "X180",
                                       phase=self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):
        # Extract data
        result = fetching_tool(self.job, data_list=["ss"], mode='live')

        # Live Plotting
        fig, ax = plt.subplots()
        fig.suptitle(f"{self.element} Ramsey Frequency Measurement")
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Excited state probability')
        (line,) = ax.plot(self.t_vec, [0.5]*len(self.t_vec), "o", color="darkblue", lw=1)
        (fit_line,) = ax.plot(self.t_vec, [0.5]*len(self.t_vec), "-", color="darkblue", lw=1)
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

        populations = None
        while result.is_processing():
            populations = result.fetch_all()[0]
            self.data.update({'populations': populations})

            # Plot Data
            line.set_data(self.t_vec, populations)
            ax.relim()
            ax.autoscale_view(scaley=False)
            plt.pause(0.05)

            # Fit Ramsey
            try:
                # Adaptive initial guess
                amplitude_guess = (np.max(populations) - np.min(populations)) / 2
                offset_guess = np.mean(populations[-10:])
                phase_guess = 90

                # Estimate frequency via FFT:
                fft_freqs = np.fft.fftfreq(len(self.t_vec), d=(self.t_vec[1] - self.t_vec[0]))
                fft_signal = np.fft.fft(populations - offset_guess)
                positive_freqs = fft_freqs[fft_freqs > 0]
                positive_fft_signal = fft_signal[fft_freqs > 0]

                dominant_freq = positive_freqs[np.argmax(np.abs(positive_fft_signal))]

                initial_guess = [amplitude_guess, dominant_freq, phase_guess, offset_guess]
                popt, pcov = curve_fit(cosine, self.t_vec, populations, p0=initial_guess, maxfev=5000)
                self.data['fit'] = {
                    'popt': popt,
                    'pcov': pcov
                }

                # ns time resolution
                fit_frequency = popt[1] * u.GHz
                freq_err = np.sqrt(pcov[1][1]) * u.GHz
                delta_MHz = ((self.ramsey_detuning- fit_frequency)/u.MHz, freq_err/u.MHz)

                self.data['delta'] = delta_MHz
                t_vec_fit = np.linspace(self.t_vec[0], self.t_vec[-1], 200)
                fit_line.set_data(t_vec_fit, cosine(t_vec_fit, *popt))
                ax.set_title(f"Measued Detuning: {delta_MHz[0]: 0f} +- {delta_MHz[1]:0f} MHz")
                ax.relim()
                ax.autoscale_view(scaley=False)
                plt.pause(0.05)
            except:
                pass
            plt.pause(0.01)
        plt.savefig(self.iname, bbox_inches="tight")
        print(f"Bare Frequency: {self.meta_dict['IF_freq']/u.MHz + self.data['delta'][0]}")