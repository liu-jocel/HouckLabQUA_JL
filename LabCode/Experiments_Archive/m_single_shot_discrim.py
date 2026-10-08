from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait, measure_3q, measure_2q
from LabCode2.Helpers.ss_helpers import find_blob_mean, find_threshold, concatenate, find_state, Discrim2Q2L
from LabCode2.Helpers.fit_functions import exponential
from LabCode2.Helpers.gate_macros import flat_top_iSWAP,flat_top_iSWAP_dg
from LabCode2.Helpers.ss_helpers import Discrim2Q2L, Discrim2Q3L, DiscrimC2Q3L, Discrim_G_C2Q3L, DiscrimCL
import pickle
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters

class SingleShotDiscrim(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element_1,element_2,element_c,shots,state_names,POVM, discrim_class,outer_folder='',path='SingleShotCal', suffix='2Q2L', meta_dict=None, plot=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)#element_c added when measuring coupler

        self.element_1 = element_1
        self.element_2 = element_2
        self.element_c = element_c
        self.shots = shots
        self.plot = plot

        #self.mean_array = discrim_class.calib_params


        self.state_names = state_names

        self.POVM = POVM

        self.elements = [element_1, element_2, element_c]

        self.job = self.acquire(self.make_prog())
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
            Ic = declare(fixed)
            Qc = declare(fixed)

            for s in self.state_names:

                I_stream_1 = declare_stream()
                Q_stream_1 = declare_stream()
                I_stream_2 = declare_stream()
                Q_stream_2 = declare_stream()
                I_stream_c = declare_stream()
                Q_stream_c = declare_stream()

                with for_(n, 0, n < shots, n + 1):
                    wait(reset_ticks, q1)
                    align()
                    self.prep_macro(s)
                    # align()#added later
                    # wait(1750, q1)#added later
                    align()
                    I1, Q1, I2, Q2, Ic, Qc = measure_3q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], self.meta_dict[self.element_c],
                                                I1, Q1, I2, Q2, Ic, Qc) #for 3q measure

                    # I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2],
                    #                                     I1, Q1, I2, Q2)

                    save(I1, I_stream_1)
                    save(Q1, Q_stream_1)
                    save(I2, I_stream_2)
                    save(Q2, Q_stream_2)
                    save(Ic, I_stream_c)
                    save(Qc, Q_stream_c)

                with stream_processing():

                    I_stream_1.save_all('I1_'+s)
                    Q_stream_1.save_all('Q1_'+s)
                    I_stream_2.save_all('I2_'+s)
                    Q_stream_2.save_all('Q2_'+s)
                    I_stream_c.save_all('Ic_'+s)
                    Q_stream_c.save_all('Qc_'+s)


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
            Ic = res_handles.get('Ic_' + s).fetch_all()['value']
            Qc = res_handles.get('Qc_' + s).fetch_all()['value']

            IQ_dict[s] = {'IQ1': I1 + 1j*Q1,
                          'IQ2': I2 + 1j*Q2,
                          'IQc': Ic + 1j*Qc,
                          }

        POVM_dict = {}
        for p in self.POVM.keys():
            IQ1 = np.zeros(0)
            IQ2 = np.zeros(0)
            IQc = np.zeros(0)

            for s in self.POVM[p]:
                IQ1 = concatenate(IQ1, IQ_dict[s]['IQ1'])
                IQ2 = concatenate(IQ2, IQ_dict[s]['IQ2'])
                IQc = concatenate(IQc, IQ_dict[s]['IQc'])

            POVM_dict[p] = {'IQ1': IQ1,
                            'IQ2': IQ2,
                            'IQc': IQc}
        self.data = POVM_dict



        if self.plot:
            fig, ax = plt.subplots(2, 2, constrained_layout=True)

            for p in POVM_dict.keys():
                ax[0][0].plot(np.imag(POVM_dict[p]['IQ1']), np.real(POVM_dict[p]['IQ1']), '.', label=p)
                ax[0][1].plot(np.imag(POVM_dict[p]['IQ2']), np.real(POVM_dict[p]['IQ2']), '.', label=p)
                ax[1][0].plot(np.imag(POVM_dict[p]['IQc']), np.real(POVM_dict[p]['IQc']), '.', label=p)

            ax[0][0].legend()
            ax[0][1].legend()
            ax[1][0].legend()
            
            ax[0][0].set_title(f'{self.element_1}')
            ax[0][1].set_title(f'{self.element_2}')
            # ax[1][0].set_title(f'{self.element_c}')

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
    outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
    pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)
    # meta_dict['qc']['read_amp'] = 0
    meta_dict['q1']['read_amp'] = 0
    Q1 = meta_dict['q1']
    Q2 = meta_dict['q2']
    Qc = meta_dict['qc']

    LOs = [Q1['q_LO'], Q2['q_LO'], Qc['q_LO']]  # [Q1['q_LO'],Q2['q_LO'], Q2['r_LO']]

    state_names= {'000', '010', '020',}
    POVM= {'q2_in_0': ['000'],
               'q2_in_1': ['010'],
               'q2_in_2': ['020'],}

    # set_LO_parameters(LOs)

    SS = SingleShotDiscrim('q1', 'q2', 'qc', 1000, state_names, POVM, Discrim_G_C2Q3L,
                           meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                           suffix='discrim')  # 'qc' added shots_SS[q]
    discrim_2Q = SS.discrim
