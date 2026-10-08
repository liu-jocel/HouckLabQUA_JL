from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.ss_helpers import find_blob_mean, find_threshold, concatenate, find_state, Discrim2Q2L
from LabCode.Helpers.ss_helpers import Discrim2Q3L


class SingleShotArb2Q(ExperimentClass):
    """
    Calibrates Arbitrary SS Discriminator for a 2 qubit system defined by element_1 and element_2.
    Required Parameters:
        shots [int]:  # of data points collected for SS Calibration
        state_names [dict]: strings of states to prepare for the discriminator in the form 'ab', for a,b in {0,1,2}
        POVM [dict]:    Dictionary that assigns collections of states to one POVM:
                e.g. POVM = {'ComputationalSubspace': ['00', '01', '10', '11],
                             'LeakageSubspace': ['02', '20', '22']
                            }
                identifies computational vs leakage subspaces

    """

    def __init__(self, element_1, element_2, shots, state_names, POVM, outer_folder='',
                 path='SingleShotCal', suffix='2QARB', meta_dict=None, config=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict)

        self.element_1 = element_1
        self.element_2 = element_2
        self.shots = shots
        self.plot = plot
        self.meta_dict = meta_dict
        self.config = config

        self.state_names = state_names

        self.POVM = POVM

        self.elements = [element_1, element_2]

        self.job = self.acquire(self.make_prog(), self.config)
        self.POVM_dict = self.analyze()
        self.discrim = discrim_class(self.POVM_dict, self.POVM)

    def prep_macro(self, state):
        print(state)
        for i in range(3):
            if state[i] == '1':
                play('X180', self.elements[i])
                # align()
            if state[i] == '2':
                play('X180', self.elements[i])
                align()
                update_frequency(self.elements[i], self.meta_dict[self.elements[i]]['f_state_IF'])
                play('X180_2', self.elements[i])
                update_frequency(self.elements[i], self.meta_dict[self.elements[i]]['q_IF'])
                align()

    def make_prog(self):

        q1 = self.meta_dict[self.element_1]['q_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            for s in self.state_names:
                I_stream_1 = declare_stream()
                Q_stream_1 = declare_stream()
                I_stream_2 = declare_stream()
                Q_stream_2 = declare_stream()

                with for_(n, 0, n < shots, n + 1):
                    wait(reset_ticks, q1)
                    align()
                    self.prep_macro(s)
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2],
                                                        I1, Q1, I2, Q2)

                    save(I1, I_stream_1)
                    save(Q1, Q_stream_1)
                    save(I2, I_stream_2)
                    save(Q2, Q_stream_2)

                with stream_processing():
                    I_stream_1.save_all('I1_' + s)
                    Q_stream_1.save_all('Q1_' + s)
                    I_stream_2.save_all('I2_' + s)
                    Q_stream_2.save_all('Q2_' + s)
        return prog

    def analyze(self):

        job = self.job

        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        IQ_dict = {}

        for s in self.state_names:
            I1 = res_handles.get('I1_' + s).fetch_all()['value']
            Q1 = res_handles.get('Q1_' + s).fetch_all()['value']
            I2 = res_handles.get('I2_' + s).fetch_all()['value']
            Q2 = res_handles.get('Q2_' + s).fetch_all()['value']

            IQ_dict[s] = {'IQ1': I1 + 1j * Q1,
                          'IQ2': I2 + 1j * Q2,
                          }

        POVM_dict = {}
        for p in self.POVM.keys():
            IQ1 = np.zeros(0)
            IQ2 = np.zeros(0)

            for s in self.POVM[p]:
                IQ1 = concatenate(IQ1, IQ_dict[s]['IQ1'])
                IQ2 = concatenate(IQ2, IQ_dict[s]['IQ2'])

            POVM_dict[p] = {'IQ1': IQ1,
                            'IQ2': IQ2,
                            }
        self.data = POVM_dict

        if self.plot:
            fig, ax = plt.subplots(2, 2, constrained_layout=True)

            for p in POVM_dict.keys():
                ax[0][0].plot(np.imag(POVM_dict[p]['IQ1']), np.real(POVM_dict[p]['IQ1']), '.', label=p)
                ax[0][1].plot(np.imag(POVM_dict[p]['IQ2']), np.real(POVM_dict[p]['IQ2']), '.', label=p)

            ax[0][0].legend()
            ax[0][1].legend()
            ax[1][0].legend()

            ax[0][0].set_title(f'{self.element_1}')
            ax[0][1].set_title(f'{self.element_2}')
            ax[0][0].set_ylabel('I')
            ax[1][0].set_xlabel('Q')
            ax[1][0].set_ylabel('I')
            ax[1][1].set_xlabel('Q')

            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)

        return IQ_dict

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data)


if __name__ == '__main__':
    pass
