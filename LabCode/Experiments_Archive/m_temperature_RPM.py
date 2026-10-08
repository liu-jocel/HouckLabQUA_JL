from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.loops import from_array

from LabCode.Helpers.fit_functions import cosine
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class TemperatureRPM(ExperimentClass):
    """
    Rabi experiement
    """
    def __init__(self, element, a_min, a_max, num_a, shots, calib_params, anharm, outer_folder='', path='Rabi', suffix='data', meta_dict=None, plot=True, confusion=np.eye(3), save=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.element = element #meta dict element name
        self.pi_amp_norm = self.meta_dict['pi_amp_ef']
        self.a_min = a_min
        self.a_max = a_max
        self.num_a = num_a
        self.amp_vec = np.linspace(self.a_min, self.a_max, self.num_a) #Raw amplitude array made from input parameters
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.freq_01 = self.meta_dict['q_IF']
        self.freq_12 = self.meta_dict['f_state_IF']
        self.anharm = anharm
        self.confusion = confusion
        self.job = self.acquire(self.make_prog())
        self.analyze()
        if save:
            self.pickle_data()
    def make_prog(self):

        qubit = self.meta_dict['q_name']
        reset_time = self.meta_dict['reset_time']
        amp_vec = self.amp_vec/self.pi_amp_norm #normalize to current amplitude for pi pulse
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            a = declare(int)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(a, self.amp_vec_norm)):
                    #case 1
                    # Rabi Thermal e population to f
                    wait(reset_time*u.ns, qubit)
                    update_frequency(qubit, self.freq_12)
                    play('X180_ef' * amp(a), qubit)
                    align()
                    update_frequency(qubit, self.freq_01)
                    # pi pulse g to e so we have contrast between f and g
                    play('X180', qubit)
                    align()
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream_0)
                    save(Q, Q_stream_0)


                    wait(reset_time * u.ns, qubit)
                    update_frequency(qubit, self.freq_01)
                    play('X180', qubit)
                    align()
                    update_frequency(qubit, self.freq_12)
                    play('X180_ef' * amp(a), qubit)
                    align()
                    align()
                    update_frequency(qubit, self.freq_01)
                    play('X180', qubit)
                    align()
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

        I_0 = I_handle_0.fetch_all()['value'].reshape(self.shots, self.num_a)
        Q_0 = Q_handle_0.fetch_all()['value'].reshape(self.shots, self.num_a)
        I_1 = I_handle_1.fetch_all()['value'].reshape(self.shots, self.num_a)
        Q_1 = Q_handle_1.fetch_all()['value'].reshape(self.shots, self.num_a)

        IQ_0 = np.mean(I_0, 0) ** 2 + np.mean(Q_0, 0) ** 2
        IQ_1 = np.mean(I_1, 0) ** 2 + np.mean(Q_1, 0) ** 2

        # Normalize data
        IQ_0 = (IQ_0-IQ_0.min())/(IQ_1.max()-IQ_1.min())
        IQ_1 = (IQ_1-IQ_1.min())/(IQ_1.max()-IQ_1.min())

        self.data = [self.amp_vec, IQ_0, IQ_1]

        popt0, cov0 = super().analyze("cosine", "RPM_seed", IQ_0, qubit=self.element)
        popt1, cov1 = super().analyze("cosine", "RPM_seed", IQ_1, qubit=self.element)

        a0 = popt0[0]
        a1 = popt1[0]

        kb = 20.83661912    # MHz/mK
        self.temp = 2*np.pi * self.anharm/(np.log((a1/a0)) * kb)



        if self.plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.amp_vec, IQ_0, 'o', label ="Experiment 0")
            ax.plot(self.amp_vec, IQ_1, 'o', label= "Experiment 1")
            try:
                ax.plot(self.amp_vec, cosine(self.amp_vec, *popt0), '-', label ="Fit 0")
                ax.plot(self.amp_vec, cosine(self.amp_vec, *popt1), '-', label="Fit 1")
            except:
                pass
            ax.set_title(f'Qubit {self.element} temperature: {self.temp} mK')
            ax.set_xlabel('Amp (arb.)')
            ax.set_ylabel('IQ amp (arb.)')
            ax.legend()
            #ax.set_title(self.element['q_name']+', T1 = ' + T1 + '+/-' + str(np.round(T1_err,4)) + ' s')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

        # job = self.job
        # shape = [self.shots, len(self.amp_vec)]
        # SS_0 = self.job_to_shots_1q_3state(job, shape,
        #                                        self.calib_params, I_name="I_0", Q_name="Q_0")
        # SS_1 = self.job_to_shots_1q_3state(job, shape,
        #                                        self.calib_params, I_name="I_1", Q_name="Q_1")
        # self.data["SS_0"] = SS_0
        # self.data["SS_1"] = SS_1
        #
        # # SS_0_normalized = 2*(SS_0-np.mean(SS_0))/(SS_0.max() - SS_0.min())
        # # SS_1_normalized = 2*(SS_1-np.mean(SS_1))/(SS_1.max() - SS_1.min())
        # # seed_1 = RPM_seed(self.amp_vec, SS_1_normalized)
        # # try:
        # #     seed_0 = RPM_seed(self.amp_vec, SS_0_normalized)
        # # except:
        # #     seed_0 = RPM_seed(self.amp_vec, SS_1_normalized)
        # #     seed_0[0] = 2*SS_1_normalized.max()-SS_1_normalized.min()
        # # popt0, pcov0 = curve_fit(sine, self.amp_vec, SS_0, p0=seed_0)
        # # popt1, pcov1 = curve_fit(sine, self.amp_vec, SS_1_normalized, p0=seed_1)
        # #
        # # print(popt0)
        # # print(popt1)
        # # a0 = popt0[0]
        # # a1 = popt1[0]
        # #
        # # kb = 20.83661912    # MHz/mK
        # # self.data['temperature'] = 2*np.pi * self.meta_dict['anharmonicity']/(np.log((a1/a0)) * kb)
        # # print(f"Temperature = {self.data['temperature']}")
        #
        #
        # if self.plot:
        #     fig, ax = plt.subplots(1, 1, constrained_layout=True)
        #     ax.plot(self.amp_vec*self.ef_amp_norm, np.mean(SS_0, axis=0), 'o', label = "Experiment 0")
        #     ax.plot(self.amp_vec*self.ef_amp_norm, np.mean(SS_1, axis=0), 'o', label = "Experiment 1")
        #     try:
        #         fit_x = np.linspace(self.amp_vec[0], self.amp_vec[-1], 200)
        #         ax.plot(self.amp_vec*self.ef_amp_norm, sine(fit_x, *popt0), '-', label = "Fit 0")
        #         ax.plot(self.amp_vec*self.ef_amp_norm, sine(fit_x, *popt1), '-', label = "Fit 1")
        #     except:
        #         pass
        #     # ax.set_title(f'Qubit {self.element} temperature: {self.data["temperature"]} mK')
        #     ax.set_xlabel('Amp (arb.)')
        #     ax.set_ylabel('IQ amp (arb.)')
        #     ax.legend()
        #     plt.savefig(self.iname, bbox_inches="tight")
        #     plt.close(fig)
