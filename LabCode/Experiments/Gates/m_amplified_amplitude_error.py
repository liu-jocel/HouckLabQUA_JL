from qm.qua import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from datetime import datetime
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.pulse_macros import *
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class AAE(ExperimentClass):
    """
    Amplified Amplitude Error Experiment implement from S. Sheldon et. al:
    DOI: https://doi.org/10.1103/PhysRevA.93.012301


    """
    def __init__(self,element, N_vec, shots, calib_params, pulse_type="X90",outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)
        self.config = config
        self.element = element #meta dict element
        self.N_vec = N_vec
        self.shots = shots
        self.pulse_type = pulse_type.upper()
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            'shots': self.shots,
            'pulse_type': pulse_type,
            'meta_dict': self.meta_dict
        }

        if pulse_type[-2:] == "80":
            self.job, self.qm = self.acquire(self.make_prog_180(), self.config, return_qm=True)
        elif pulse_type[-2:] == "90":
            self.job, self.qm = self.acquire(self.make_prog_90(), self.config, return_qm=True)
        else:
            raise ValueError("pulse_type needs to be X90, Y90, X180, or Y180")
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog_90(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            m = declare(int)
            num_pulse = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            n_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                reset_frame(qubit)
                save(n, n_stream)
                with for_each_(num_pulse, self.N_vec):
                    play_pulse(qubit, self.pulse_type,
                               phase=self.meta_dict[self.pulse_type]['phase'])
                    with for_(m, 0, m < num_pulse, m + 1):
                        play_pulse(qubit, self.pulse_type,
                                   phase=self.meta_dict[self.pulse_type]['phase'])
                        play_pulse(qubit, self.pulse_type,
                                   phase=self.meta_dict[self.pulse_type]['phase'])

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
                            play_pulse(qubit, "X180",
                                       phase=self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
                    #wait(3 * u.ms, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.N_vec)).average().save("ss")
                n_stream.save('n')
        return prog
    def make_prog_180(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            m = declare(int)
            num_pulse = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()
            n_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(num_pulse, self.N_vec):
                    # Pi/2 Rotation
                    play_pulse(qubit, self.pulse_type[0]+"90",
                               phase=self.meta_dict[self.pulse_type[0]+"90"]['phase'])
                    with for_(m, 0, m < num_pulse, m + 1):
                        play_pulse(qubit, self.pulse_type,
                                   phase=self.meta_dict[self.pulse_type]['phase'])

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
                            play_pulse(qubit, "X180",
                                       phase=self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                ss_stream.buffer(len(self.N_vec)).average().save("ss")
                n_stream.save('n')
        return prog

    def analyze(self, job, qm):
        result = fetching_tool(self.job, data_list=["ss", "n"], mode='live')

        # Live Plotting
        fig, ax = plt.subplots()
        ax.set_xlabel('N')
        ax.set_ylabel('Ground state probability')
        ax.set_ylim(0,1)
        (line,) = ax.plot(self.N_vec, [0.5] * len(self.N_vec), "o", color="darkblue", lw=1)
        (fit_line,) = ax.plot(self.N_vec, [0.5] * len(self.N_vec), "-", color="darkblue", lw=1)
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figures

        while result.is_processing():
            populations = 1-np.array(result.fetch_all()[0])
            self.data.update({'populations': populations})

            # Plot Data
            line.set_data(self.N_vec, populations)

            # fitting
            try:
                initial_guess = [0, 0.1]
                popt, pcov = curve_fit(fit_AAE_X90, self.N_vec, populations, p0=initial_guess, maxfev=5000)
                self.data['fit'] = {
                    'popt': popt,
                    'pcov': pcov,
                }

                e = popt[0]
                sigma_e = np.sqrt(pcov[0, 0])

                amplitude_error = 2*e/np.pi * self.meta_dict[self.pulse_type]['amp']
                amplitude_error_err = 2 * abs(sigma_e)/np.pi * self.meta_dict[self.pulse_type]['amp']
                self.data['amplitude_error'] = amplitude_error

                # plot fit
                fit_line.set_data(self.N_vec, fit_AAE_X90(self.N_vec, *popt))
                fig.suptitle(f"{self.element} {self.pulse_type} Amplitude Error: {amplitude_error: 0.8f} ± {amplitude_error_err: 0.8f}")
            except:
                pass
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)
        print("")  # flush console
        print(f"{self.element} New {self.pulse_type}  amp: {self.meta_dict[self.pulse_type]['amp'] + amplitude_error} ± {amplitude_error_err}")
        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

def fit_AAE_X90(n, e, a):
    return a + 0.5 * (-1)**n * np.cos(np.pi/2 - 2*n*e)