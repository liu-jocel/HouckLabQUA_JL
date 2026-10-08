from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("qtagg", force=True)
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

def cosine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset



class QubitSpecVsFlux(ExperimentClass):
    """
    Qubit spectroscopy versus flux
    """
    def __init__(self,element,f_vec, dc_vec, shots, outer_folder='',path='trans', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.f_vec = f_vec
        self.dc_vec = dc_vec
        self.shots = shots
        self.config = config
        self.resonator_curve = cosine_func(self.dc_vec, *self.meta_dict['resonator_fit_parameters']).astype(int)
        self.job, qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            'dc_vec': self.dc_vec,
            'shots': self.shots,
            "meta_dict": self.meta_dict
        }

        # Analyze and save data
        job, qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(job, qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        flux_line = self.meta_dict['flux_name']
        flux_settle_time = self.meta_dict['flux_settle_time']
        resonator = self.meta_dict['r_name']

        with program() as prog:
            n = declare(int)  # QUA variable for the averaging index
            f = declare(int)  # QUA variable for the qubit frequency
            dc = declare(fixed)  # QUA variable for the flux dc level
            I = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            resonator_freq = declare(int, value=self.resonator_curve)  # res freq vs flux table
            index = declare(int, value=0)  # index to get the right resonator freq for a given flux
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(f, self.f_vec)):
                    # Update the qubit frequency
                    update_frequency(qubit, f)
                    assign(index, 0)
                    with for_(*from_array(dc, self.dc_vec)):
                        # Update the resonator frequency to always measure on resonance
                        update_frequency(resonator, resonator_freq[index])
                        # Flux sweeping
                        set_dc_offset(flux_line, "single", dc)
                        wait(flux_settle_time * u.ns, flux_line, resonator, qubit)
                        # Play a qubit pulse on the qubit
                        play("constant_tone", qubit)
                        # Align the two elements to measure after playing the qubit pulse.
                        # One can also measure the resonator while driving the qubit by commenting the 'align'
                        align(qubit, resonator)
                        # Measure the state of the resonator
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        # Wait for the qubit to decay to the ground state
                        wait(reset_time * u.ns, flux_line, resonator, qubit)
                        # Update the resonator frequency vs flux index
                        assign(index, index + 1)
                        # Save the 'I' & 'Q' quadratures to their respective streams
                        save(I, I_st)
                        save(Q, Q_st)
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
                I_st.buffer(len(self.dc_vec)).buffer(len(self.f_vec)).average().save("I")
                Q_st.buffer(len(self.dc_vec)).buffer(len(self.f_vec)).average().save("Q")
                n_st.save("iteration")
        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

        # Live Plotting
        fig = plt.figure()
        plt.suptitle(f"Qubit spectroscopy, spec_amp={round(self.meta_dict['cw_amp'],3)}V")
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        def if_to_qubit_freq_GHz(if_freq):
            return (self.meta_dict['q_LO']['LO_freq'] + if_freq*u.MHz)/u.GHz
        def qubit_freq_to_if_MHz(qubit_freq):
            return (self.meta_dict['q_LO']['LO_freq'] - qubit_freq*u.GHz)/u.MHz
        while results.is_processing():
            # Fetch results
            I, Q, iteration = results.fetch_all()
            # Progress Bar
            progress_counter(iteration, self.shots, progress_bar=True, percent=True, start_time=results.get_start_time())
            # Convert results into Volts and normalize
            S = u.demod2volts(I + 1j * Q, self.meta_dict['read_len'])
            R = 20 * np.log10(np.abs(S))  # Amplitude, dBm
            phase = np.angle(S)  # Phase
            # 2D spectroscopy plot
            ax1 = plt.subplot(211)
            ax1.cla()
            ax1.set_title(r"Magnitude [dBm]")
            pcm1 = ax1.pcolor(self.dc_vec, self.f_vec/u.MHz, R)
            ax1.set_ylabel("IF Frequency [MHz]")
            ax1.set_xlabel("Voltage [V]")
            secax1 = ax1.secondary_yaxis('right', functions=(if_to_qubit_freq_GHz, qubit_freq_to_if_MHz))
            secax1.set_ylabel("Qubit Frequency [GHz]")
            ax2 = plt.subplot(212)
            ax2.cla()
            ax2.set_title("Phase [rad]")
            pcm2 = ax2.pcolor(self.dc_vec, self.f_vec/u.MHz, R)
            ax2.set_ylabel("IF Frequency [MHz]")
            ax2.set_xlabel("Voltage [V]")
            secax2 = ax2.secondary_yaxis('right', functions=(if_to_qubit_freq_GHz, qubit_freq_to_if_MHz))
            secax2.set_ylabel("Qubit Frequency [GHz]")
            plt.pause(0.5)
            plt.tight_layout()

            self.data.update({
                "IQ_mag": R,
                "IQ_phase": phase
            })
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        qm.close()

        # # Find the resonator frequency vs flux minima
        # minima = np.zeros(len(self.dc_vec))
        # for i in range(len(self.dc_vec)):
        #     minima[i] = self.f_vec[np.argmin(R.T[i])]
        # # Cosine fit
        # amp_guess = abs(max(minima) - min(minima))
        # freq_guess = 1 / abs(self.dc_vec[-1] - self.dc_vec[0])
        # offset_guess = np.mean(minima)
        # initial_guess = [amp_guess, freq_guess, 0, offset_guess]  # Initial guess for the parameters
        # fit_params, _ = curve_fit(cosine_func, self.dc_vec, minima, p0=initial_guess, maxfev=10000)
        # # Get the fitted values
        # amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
        # print("fitting parameters", fit_params)
        # # Generate the fitted curve using the fitted parameters
        # fitted_curve = cosine_func(self.dc_vec, amplitude_fit, frequency_fit, phase_fit, offset_fit)
        #
        # plt.figure()
        # plt.suptitle(f"Resonator spectroscopy - LO = {self.meta_dict['r_LO']['LO_freq'] / u.GHz} GHz")
        # plt.pcolor(self.dc_vec, self.f_vec / u.MHz, R)
        # plt.plot(self.dc_vec, minima / u.MHz, "x-", color="red", label="Flux minima")
        # plt.plot(self.dc_vec, fitted_curve / u.MHz, label="Fitted Cosine", color="black")
        # plt.xlabel("Flux bias [V]")
        # plt.ylabel("Readout IF [MHz]")
        # # plt.legend()



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])