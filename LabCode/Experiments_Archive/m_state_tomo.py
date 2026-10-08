from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_2q
from LabCode2.Helpers.MLE import findRhoMLE
from LabCode2.Helpers.SPAM_correction import correct_counts_array
import qiskit
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import StateTomography as StateTomographyQiskit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis
from LabCode2.Helpers.iSWAP_tomography import plot_statetomo

backend = AerSimulator.from_backend(FakeParis())

class StateTomography(ExperimentClass):
    """
    State tomography experiement
    """
    def __init__(self,element_1,element_2,coupler,circuit_macro,macro_args,shots,calib_params,duration_ns=0,outer_folder='',path='StateTomography',
                 suffix='data', meta_dict=None, plot=True, confusion = np.identity(4),run=True,circ=qiskit.QuantumCircuit(2),**kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)

        self.__dict__.update(kwargs)
        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.coupler = coupler
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion
        self.circuit_macro = circuit_macro
        self.macro_args = macro_args
        self.duration_ns = duration_ns
        self.circ = circ

        if run:
            self.job = self.acquire(self.make_prog())
            self.analyze()


    def make_prog(self):

        q1 = self.meta_dict[self.element_1]['q_name']
        q2 = self.meta_dict[self.element_2]['q_name']
        c = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        shots = self.shots

        # Qiskit ordering
        meas_pulses = [('Zero', 'Zero'),
                        ('Zero', 'Ym90'),
                        ('Zero', 'X90'),
                        ('Ym90', 'Zero'),
                        ('Ym90', 'Ym90'),
                        ('Ym90', 'X90'),
                        ('X90', 'Zero'),
                        ('X90', 'Ym90'),
                        ('X90', 'X90')]

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                for m in meas_pulses:
                    wait_for_trigger('q1')
                    align()
                    # reset_phase(q1)
                    # reset_phase(q2)
                    # reset_phase(c)
                    wait(reset_ticks, q1)
                    # reset_frame(q1,q2,c)

                    align()
                    self.circuit_macro(self.element_1, self.element_2, self.coupler, self.macro_args)
                    align()
                    play(m[0], q1)
                    play(m[1], q2)
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1, I2, Q2)

                    save(I1, I1_stream)
                    save(Q1, Q1_stream)
                    save(I2, I2_stream)
                    save(Q2, Q2_stream)

                    # reset_frame(q1,q2)

                    align()

            with stream_processing():
                I1_stream.save_all("I_c1")
                Q1_stream.save_all("Q_c1")
                I2_stream.save_all("I_c2")
                Q2_stream.save_all("Q_c2")

        return prog

    def analyze(self):

        counts_array = super().job_to_shots_2q(self.job, [self.shots, 9], self.calib_params)
        counts_array = correct_counts_array(self.confusion,counts_array)

        backend = AerSimulator.from_backend(FakeParis())

        tomo_qiskit = StateTomographyQiskit(self.circ)
        analysis_qiskit = tomo_qiskit.analysis
        tomo_init_data = tomo_qiskit.run(backend, seed_simulation=100).block_for_results()

        raw_data = np.transpose(counts_array)

        data_holder = []

        for i in range(9):
            data_holder.append(
                {'counts': {'00': raw_data[i][0], '01': raw_data[i][1], '10': raw_data[i][2], '11': raw_data[i][3]},
                 'metadata': tomo_init_data.data(i)['metadata']})

        data_holder_qiskit = ExperimentData()
        data_holder_qiskit.add_data(data_holder)

        fitted_data = analysis_qiskit.run(data_holder_qiskit).analysis_results('state').value

        print(fitted_data.purity())
        self.purity = np.real(fitted_data.purity())

        self.data = [raw_data,fitted_data]
        if self.plot:
            plot_statetomo(self, self.data[1],duration=self.duration_ns)


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])