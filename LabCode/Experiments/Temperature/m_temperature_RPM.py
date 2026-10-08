from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from LabCode.Helpers.fit_functions import sine, RPM_seed
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class TemperatureRPM(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self, element, amp_vec, shots, outer_folder='', path='Rabi', suffix='data', config = None, meta_dict=None, plot=True, confusion=np.eye(3), save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.ef_amp_norm = self.meta_dict['ef_amp']
        self.amp_vec = amp_vec
        self.shots = shots
        self.plot = plot
        self.freq_01 = self.meta_dict['q_IF']
        self.freq_12 = self.meta_dict['f_state_IF']
        self.confusion = confusion
        self.config = config
        self.data = {
            'qubit': self.element,
            "amp_vec": self.amp_vec,
            "ef_amp_max": self.ef_amp_norm,
            "meta_dict": self.meta_dict
        }
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        amp_vec = self.amp_vec
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            a = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(a, amp_vec)):
                    wait(reset_time*u.ns, qubit)
                    #case 1
                    # Rabi Thermal e population to f
                    update_frequency(qubit, self.freq_12)
                    play('X180_ef' * amp(a), qubit)
                    update_frequency(qubit, self.freq_01)
                    # pi pulse g to e so we have contrast between f and g
                    play('X180', qubit)
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_0)
                    save(Q, Q_stream_0)

                    wait(reset_time * u.ns, qubit)
                    # case 2
                    # Rabi thermal g to f
                    update_frequency(qubit, self.freq_01)
                    play('X180', qubit)
                    update_frequency(qubit, self.freq_12)
                    play('X180_ef' * amp(a), qubit)
                    update_frequency(qubit, self.freq_01)
                    play('X180', qubit)
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_1)
                    save(Q, Q_stream_1)



            with stream_processing():
                I_stream_0.save_all("I_0")
                Q_stream_0.save_all("Q_0")
                I_stream_1.save_all("I_1")
                Q_stream_1.save_all("Q_1")
        return prog

    def analyze(self):
        job = self.job

        I_handle_0 = job.result_handles.get("I_0")
        Q_handle_0 = job.result_handles.get("Q_0")
        I_handle_1 = job.result_handles.get("I_1")
        Q_handle_1 = job.result_handles.get("Q_1")

        I_handle_0.wait_for_all_values()
        Q_handle_0.wait_for_all_values()
        I_handle_1.wait_for_all_values()
        Q_handle_1.wait_for_all_values()

        I_0 = I_handle_0.fetch_all()['value'].reshape(self.shots, len(self.amp_vec))
        Q_0 = Q_handle_0.fetch_all()['value'].reshape(self.shots, len(self.amp_vec))
        I_1 = I_handle_1.fetch_all()['value'].reshape(self.shots, len(self.amp_vec))
        Q_1 = Q_handle_1.fetch_all()['value'].reshape(self.shots, len(self.amp_vec))

        self.data.update({"I_0": I_0,
                          "Q_0": Q_0,
                          "I_1": I_1,
                          "Q_1": Q_1})

        IQ_0 = np.mean(I_0, 0) ** 2 + np.mean(Q_0, 0) ** 2
        IQ_1 = np.mean(I_1, 0) ** 2 + np.mean(Q_1, 0) ** 2

        # Normalize data
        IQ_0 = 2*(IQ_0 - IQ_0.mean()) / (IQ_1.max() - IQ_1.min())
        IQ_1 = 2*(IQ_1 - IQ_1.mean()) / (IQ_1.max() - IQ_1.min())

        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.set_title(f'Qubit {self.element} mK')
            ax.plot(self.amp_vec, IQ_0, 'o', label="Experiment 0", color="black")
            ax.plot(self.amp_vec, IQ_1, 'o', label="Experiment 1", color="red")
            try:
                seed_1 = RPM_seed(self.amp_vec, IQ_1)
                try:
                    seed_0 = RPM_seed(self.amp_vec, IQ_0)
                except:
                    seed_0 = RPM_seed(self.amp_vec, IQ_1)
                    seed_0[0] = 2 * IQ_1.max() - IQ_1.min()
                bounds = ([0, 0.9 * seed_1[1], -np.pi, -np.inf],
                          [np.inf, 1.1 * seed_1[1], np.pi, np.inf])
                popt0, pcov0 = curve_fit(sine, self.amp_vec, IQ_0, p0=seed_0, bounds=bounds)
                popt1, pcov1 = curve_fit(sine, self.amp_vec, IQ_1, p0=seed_1, bounds=bounds)
                print(popt0, np.diag(pcov0)**2)
                print(popt1, np.diag(pcov1) ** 2)
                a0 = popt0[0]#/(popt0[0] + popt1[0])
                a1 = popt1[0]#/(popt0[0] + popt1[0])

                kb = 20.83661912  # MHz/mK

                qubit_freq = (self.meta_dict['q_IF'] + self.meta_dict['q_LO']['LO_freq'])/u.MHz # in MHz (or convert if needed)
                self.data['temperature'] =  qubit_freq / (np.log(a1 / a0) * kb)
                self.data['fit_0'] = popt0, pcov0
                self.data['fit_1'] = popt1, pcov1
                print(f"Temperature = {self.data['temperature']}")
                ax.set_title(f'Qubit {self.element}, Temperature: {self.data["temperature"]} mK')
                ax.plot(self.amp_vec, sine(self.amp_vec, *popt0), '-', label="Fit 0", color="black")
                ax.plot(self.amp_vec, sine(self.amp_vec, *popt1), '-', label="Fit 1", color="red")
            except:
                pass
            ax.set_xlabel('Amp (arb.)')
            ax.set_ylabel('IQ amp (arb.)')
            ax.legend()
            # ax.set_title(self.element['q_name']+', T1 = ' + T1 + '+/-' + str(np.round(T1_err,4)) + ' s')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)