from qm.qua import *
import numpy as np
from datetime import datetime
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib
matplotlib.use("qtagg", force=True)
import matplotlib.pyplot as plt
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiChevronTimedIQ(ExperimentClass):
    """
    Rabi chevron experiment reading out IQ data
    """
    def __init__(self,element, t_vec, f_vec, shots, outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.f_vec = f_vec
        self.t_vec = t_vec
        self.shots = shots

        self.read_len = self.meta_dict['global_params']['read_len']

        self.data = {
            'qubit': self.element,
            "shots": self.shots,
            "f_vec": self.f_vec,
            "t_vec": self.t_vec,
            "meta_dict": self.meta_dict
        }

        self.config = config
        job, qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(job, qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        IF_freq = self.meta_dict['q_IF']
        shots = self.shots
        with program() as prog:
            n = declare(int)  # QUA variable for the averaging loop
            f = declare(int)  # QUA variable for the qubit frequency
            t = declare(int)  # QUA variable for the qubit pulse duration
            I = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(n, 0, n < self.shots, n + 1):  # QUA for_ loop for averaging
                with for_(*from_array(t, self.t_vec)):  # QUA for_ loop for sweeping the pulse duration
                    with for_(*from_array(f, self.f_vec)):  # QUA for_ loop for sweeping the frequency
                        # Update the frequency of the digital oscillator linked to the qubit element
                        update_frequency(qubit,  f + IF_freq)
                        # Play the qubit pulse with a variable duration (in clock cycles = 4ns)
                        play("X180", qubit, duration=t)
                        # Align the two elements to measure after playing the qubit pulse.
                        align(qubit, resonator)
                        # Measure the state of the resonator.
                        # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        # Wait for the qubit to decay to the ground state
                        wait(reset_time * u.ns, resonator)
                        # Save the 'I' & 'Q' quadratures to their respective streams
                        save(I, I_st)
                        save(Q, Q_st)
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
                I_st.buffer(len(self.f_vec)).buffer(len(self.t_vec)).average().save("I")
                Q_st.buffer(len(self.f_vec)).buffer(len(self.t_vec)).average().save("Q")
                n_st.save("iteration")
            return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

        # Live Plotting
        fig = plt.figure()
        plt.suptitle(r"Qubit Timed Rabi")
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            I, Q, iteration = results.fetch_all()
            # Convert results into Volts
            S = u.demod2volts(I + 1j * Q, self.meta_dict['read_len'])
            R = 20 * np.log10(np.abs(S))  # Amplitude, dBm
            phase = np.angle(S)  # Phase
            # Progress bar
            progress_counter(iteration, self.shots, progress_bar=True, percent=True, start_time=results.get_start_time())
            # Plot results
            plt.subplot(211)
            plt.suptitle(f"Rabi chevron with LO={self.meta_dict['q_LO']['LO_freq'] / u.GHz} GHz and IF={self.meta_dict['q_IF']/ u.MHz} MHz. Pi amp: {self.meta_dict['pi_amp']} V")
            plt.cla()
            plt.title(r"Magnitude (dBm)")
            plt.pcolor(self.f_vec / u.MHz, self.t_vec * u.ns, R)
            plt.xlabel("Frequency detuning [MHz]")
            plt.ylabel("Pulse duration [ns]")
            plt.subplot(212)
            plt.cla()
            plt.title("Phase")
            plt.pcolor(self.f_vec / u.MHz, self.t_vec * u.ns, np.unwrap(phase))
            plt.xlabel("Frequency detuning [MHz]")
            plt.ylabel("Pulse duration [ns]")
            plt.tight_layout()
            plt.pause(1)
        self.data.update({
            "IQ_mag": R,
            "IQ_phase": phase
        })
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        qm.close()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])