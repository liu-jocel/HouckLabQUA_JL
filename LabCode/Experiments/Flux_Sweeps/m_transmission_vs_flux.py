from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qualang_tools.results.data_handler import DataHandler
from datetime import datetime
import sys

sys.path.append('/')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class TransmissionVsFlux(ExperimentClass):
    """
    Transmission experiement
    """

    def __init__(self, element, f_vec, dc_vec, shots, outer_folder='', path='trans', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element  # meta dict element name
        self.f_vec = f_vec
        self.dc_vec = dc_vec
        self.num_f = len(self.f_vec)
        self.shots = shots
        self.meta_dict = meta_dict
        self.config = config

        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            'dc_vec': self.dc_vec,
            "meta_dict": self.meta_dict
        }

        job, qm = self.acquire(self.make_prog(), config=self.config, return_qm=True)
        self.analyze(job, qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        flux_line = self.meta_dict['flux_name']
        flux_settle_time = self.meta_dict['flux_settle_time']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            dc = declare(fixed)  # QUA variable for the flux bias
            I_st = declare_stream()
            Q_st = declare_stream()
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(n, 0, n < shots, n + 1):
                with for_each_(f, self.f_vec):
                    update_frequency(resonator, f)
                    with for_(*from_array(dc, self.dc_vec)):
                        set_dc_offset(flux_line, "single", dc)
                        wait(flux_settle_time * u.ns, qubit, resonator, flux_line)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        wait(resonator_dump_time*u.ns)
                        save(I, I_st)
                        save(Q, Q_st)
                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.dc_vec)).buffer(len(self.f_vec)).average().save("I")
                Q_st.buffer(len(self.dc_vec)).buffer(len(self.f_vec)).average().save("Q")
                n_st.save("iteration")

        return prog
    def analyze(self, job, qm,):
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

        # Live Plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            I, Q, iteration = results.fetch_all()
            # Progress bar
            progress_counter(iteration, self.shots, progress_bar=True, percent=True,  start_time=results.get_start_time())
            # Convert results into Volts and normalize
            S = u.demod2volts(I + 1j * Q, self.meta_dict['read_len'])
            R = 20 * np.log10(np.abs(S))  # Amplitude, dBm
            phase = np.angle(S)  # Phase
            # 2D spectroscopy plot
            plt.subplot(211)
            plt.cla()
            plt.suptitle(f"Resonator spectroscopy - LO = {self.meta_dict['r_LO']['LO_freq'] / u.GHz} GHz")
            plt.title(r"Amplitude (dBm)")
            plt.pcolor(self.dc_vec, self.f_vec / u.MHz, R)
            plt.ylabel("Readout IF [MHz]")
            plt.subplot(212)
            plt.cla()
            plt.title("Phase (rad)")
            plt.pcolor(self.dc_vec, self.f_vec / u.MHz, signal.detrend(np.unwrap(phase)))
            plt.xlabel("Flux bias [V]")
            plt.ylabel("Readout IF [MHz]")
            plt.pause(0.5)
            plt.tight_layout()

            self.data.update({
                "IQ_mag": R,
                "IQ_phase": phase
            })
        qm.close()
        plt.close(fig)

        # Fitting to cosine resonator frequency response
        def cosine_func(x, amplitude, frequency, phi, offset):
            return amplitude * np.cos(2 * np.pi * frequency * x + phi) + offset

        # Find the resonator frequency vs flux minima
        minima = np.zeros(len(self.dc_vec))
        for i in range(len(self.dc_vec)):
            minima[i] = self.f_vec[np.argmin(R.T[i])]
        # Cosine fit
        amp_guess = abs(max(minima) - min(minima))
        freq_guess = 1/abs(self.dc_vec[-1]-self.dc_vec[0])
        offset_guess = np.mean(minima)
        initial_guess = [amp_guess, freq_guess, 0, offset_guess]  # Initial guess for the parameters
        fit_params, _ = curve_fit(cosine_func, self.dc_vec, minima, p0=initial_guess, maxfev=10000)
        # Get the fitted values
        amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
        print("fitting parameters", ','.join(map(str, fit_params)))
        # Generate the fitted curve using the fitted parameters
        fitted_curve = cosine_func(self.dc_vec, amplitude_fit, frequency_fit, phase_fit, offset_fit)

        plt.figure()
        plt.suptitle(f"Resonator spectroscopy - LO = {self.meta_dict['r_LO']['LO_freq'] / u.GHz} GHz")
        plt.pcolor(self.dc_vec, self.f_vec / u.MHz, R)
        plt.plot(self.dc_vec, minima / u.MHz, "x-", color="red", label="Flux minima")
        plt.plot(self.dc_vec, fitted_curve / u.MHz, label="Fitted Cosine", color="black")
        plt.xlabel("Flux bias [V]")
        plt.ylabel("Readout IF [MHz]")
        # plt.legend()

        plt.savefig(self.iname, bbox_inches="tight")

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])

