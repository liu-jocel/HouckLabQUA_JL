from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class FreqCal1Q(ExperimentClass):
    """
    Frequency Calibration of 1Q gate
    """
    def __init__(self,element, f_vec, shots, calib_params, num_pi_pulses=1, pulse_type= "X180", outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.pulse_type = pulse_type
        self.f_vec = f_vec

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
            "meta_dict": self.meta_dict
        }

        self.config = config
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        IF_freq = self.meta_dict['IF_freq']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        if self.pulse_type == "X180":
            with program() as prog:
                n = declare(int)
                df = declare(int)
                m = declare(int)
                state = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                I_reset = declare(fixed)
                ss_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_each_(df, self.f_vec):
                        update_frequency(qubit, df+IF_freq)
                        with for_(m, 0, m < self.num_pi, m + 1):
                            play('X180', qubit)
                            play('X180' * amp(-1), qubit)

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
                        wait(resonator_dump_time * u.ns, qubit)

                        # Active reset
                        update_frequency(qubit, IF_freq)
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)

                with stream_processing():
                    ss_stream.buffer(len(self.f_vec)).average().save("ss")
            return prog

        if self.pulse_type == "X90":
            with program() as prog:
                n = declare(int)
                df = declare(int)
                m = declare(int)
                state = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                I_reset = declare(fixed)
                ss_stream = declare_stream()
                with for_(n, 0, n < shots, n + 1):
                    with for_each_(df, self.f_vec):
                        update_frequency(qubit, df + IF_freq)
                        with for_(m, 0, m < self.num_pi, m + 1):
                            play('X90', qubit)
                            play('mX90', qubit)
                            #play('Y90', qubit)
                            #play('Ym90', qubit)

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
                        update_frequency(qubit, IF_freq)
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            wait(resonator_dump_time * u.ns, qubit)
                            assign(I_reset, I)

                with stream_processing():
                    ss_stream.buffer(len(self.f_vec)).average().save("ss")
            return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss"])
        ss_array = result.fetch_all()[0]
        self.data.update({
            "ss_data": ss_array,
            })

        IF_freq = self.meta_dict['IF_freq']
        f_vec = self.f_vec/u.MHz


        plt.suptitle(self.element + f" {self.num_pi} pulses")
        plt.subplot(111)
        plt.cla()
        try:
            def cosine_sq(x,T, A, B):
                return A*np.cos((np.pi/T)*x)**2 + B
            popt, pcov = curve_fit(cosine_sq, f_vec, ss_array, p0=[(f_vec[-1]-f_vec[0])/2, -1, 0.5], maxfev=5000)
            self.data.update({
                'fit_popt': popt, 'fit_pcov': pcov,
                'fit_func': "cosine_sq"
            })
            x_data = np.linspace(f_vec[0], f_vec[-1], 100)
            y_data = cosine_sq(x_data, *popt)
            #plt.plot(x_data, y_data)
            #plt.title(f" Freq: {x_data[np.argmin(y_data)] + IF_freq/u.MHz} MHz")
            print(f" Freq: {x_data[np.argmin(y_data)] +IF_freq/u.MHz} MHz")
        except:
            print("Fit Failed")
        plt.plot(f_vec, ss_array, "o")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Excited State Population")
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])