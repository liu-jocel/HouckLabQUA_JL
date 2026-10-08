from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from LabCode.Helpers.fit_functions import decaying_cosine, rabi_seed
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiLinecutSS(ExperimentClass):
    """
    Rabi Linecut wit single shot readout
    """
    def __init__(self,element, amp_vec, shots, calib_params, num_pi_pulses=1, pulse_type= "X180", outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.pulse_type = pulse_type
        self.amp_vec_abs = amp_vec
        self.amp_vec = self.amp_vec_abs/meta_dict[pulse_type]['amp']
        self.shots = shots
        self.num_pi = num_pi_pulses
        self.calib_params = calib_params

        self.read_len = self.meta_dict['global_params']['read_len']

        self.data = {
            'qubit': self.element,
            "pulse_type": self.pulse_type,
            "number_pulses": self.num_pi,
            "shots": self.shots,
            "amp_vec": self.amp_vec,
            "meta_dict": self.meta_dict
        }

        self.config = config
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        # if  self.pulse_type == "X180" or self.pulse_type == "Y180":
        #     with program() as prog:
        #         n = declare(int)
        #         a = declare(fixed)
        #         m = declare(int)
        #         state = declare(int)
        #         I = declare(fixed)
        #         Q = declare(fixed)
        #         I_reset = declare(fixed)
        #         ss_stream = declare_stream()
        #         with for_(n, 0, n < shots, n + 1):
        #             with for_(*from_array(a, self.amp_vec)):
        #                 with for_(m, 0, m < self.num_pi, m + 1):
        #                     play('X180' * amp(a), qubit)
        #                 update_frequency(qubit, IF_freq)
        #                 # SS Readout
        #                 align(qubit, resonator)
        #                 I, Q = measure_1q(self.meta_dict, I, Q)
        #                 assign(I_reset, I)
        #                 with if_(I > ss_thresh):
        #                     assign(state, 1)
        #                 with else_():
        #                     assign(state, 0)
        #                 save(state, ss_stream)
        #                 align(qubit, resonator)
        #
        #                 # Active reset
        #                 with while_(I_reset > ground_confidence_threshold):
        #                     with if_(I_reset > ss_thresh):
        #                         play('X180', qubit)
        #                         align(qubit, resonator)
        #                     I, _ = measure_1q(self.meta_dict, I, Q)
        #                     assign(I_reset, I)
        #                     wait(resonator_dump_time * u.ns, qubit)
        #
        #         with stream_processing():
        #             ss_stream.buffer(len(self.amp_vec)).average().save("ss")

        if self.pulse_type == "X90" or self.pulse_type == "Y90":
            with program() as prog:
                n = declare(int)
                a = declare(fixed)
                m = declare(int)
                state = declare(int)
                I = declare(fixed)
                Q = declare(fixed)
                I_reset = declare(fixed)
                ss_stream = declare_stream()
                I_stream  = declare_stream()
                Q_stream = declare_stream()

                with for_(n, 0, n < shots, n + 1):
                    with for_(*from_array(a, self.amp_vec)):
                        with for_(m, 0, m < self.num_pi, m + 1):
                            play(self.pulse_type * amp(a), qubit)
                            play(self.pulse_type * amp(a), qubit)
                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_stream)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        align(qubit, resonator)

                        # Active reset
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            wait(resonator_dump_time * u.ns, qubit)
                        # wait(2 * u.ms, qubit)

                with stream_processing():
                    ss_stream.buffer(len(self.amp_vec)).average().save("ss")
                    I_stream.buffer(len(self.amp_vec)).save_all("I")
                    Q_stream.buffer(len(self.amp_vec)).save_all("Q")
        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss", "I", "Q"])
        ss_array = result.fetch_all()[0]
        I_array = result.fetch_all()[1]
        Q_array = result.fetch_all()[2]
        self.data.update({
            "ss_data": ss_array,
            "I": I_array,
            "Q": Q_array,
            })

        plt.subplot(111)
        plt.cla()
        try:
            def cosine_sq(x,T, A, B):
                return A*np.cos((np.pi/T)*x)**2 + B
            popt, pcov = curve_fit(cosine_sq, self.amp_vec_abs, ss_array, p0=[((self.amp_vec_abs[-1]-self.amp_vec_abs[0])), 1, 0], maxfev=20000)
            self.data.update({
                'fit_popt': popt, 'fit_pcov': pcov,
                'fit_func': "cosine_sq"
            })
            x_data = np.linspace(self.amp_vec_abs[0], self.amp_vec_abs[-1], 100)
            y_data = cosine_sq(x_data, *popt)
            plt.plot(x_data, y_data)
            plt.title(f" Max Amp: {x_data[np.argmax(y_data)]} V")
            print(f" Max Amp: {x_data[np.argmax(y_data)]} V")
        except:
            print("Fit Failed")
        plt.plot(self.amp_vec_abs, ss_array, "o")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Excited State Population")
        plt.suptitle('SS Data. ' + self.element + " " + self.pulse_type + f" {(self.meta_dict['q_LO']['LO_freq'] + self.meta_dict['IF_freq'] + self.meta_dict['detuning'] ) / u.MHz} MHz. {self.num_pi} pulses")
        plt.grid()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()