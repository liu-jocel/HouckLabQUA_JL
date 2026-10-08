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



class UniTomography(ExperimentClass):
    """
    unitary tomography experiment
    """
    def __init__(self,element_1,element_2,coupler,circuit_macro,macro_args,gate_len,shots,calib_params,duration_ns=0,outer_folder='',path='StateTomography',
                 suffix='data', meta_dict=None, plot=True, confusion = np.identity(4),run=True,circ=qiskit.QuantumCircuit(2),**kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict)


        self.__dict__.update(kwargs)
        self.element_1 = element_1 #meta dict element
        self.element_2 = element_2
        self.coupler = coupler
        self.gate_len = gate_len
        self.shots = shots
        self.calib_params = calib_params
        self.plot = plot
        self.confusion = confusion
        self.circuit_macro = circuit_macro
        self.macro_args = macro_args
        self.duration_ns = duration_ns
        self.circ = circ

     #   self.iname = self.iname

        if run:
            self.job = self.acquire(self.make_prog())
            self.analyze()


    def make_prog(self):

        # q1 = self.meta_dict[self.element_1]['q_name']
        # q2 = self.meta_dict[self.element_2]['q_name']
        # c = self.meta_dict[self.coupler]['c_name']
        # reset_ticks = self.meta_dict['global_params']['reset_ticks']
        # shots = self.shots

        q1 = self.element_1['q_name']
        q2 = self.element_2['q_name']
        c = self.coupler['c_name']
        reset_ticks = self.element_1['reset_ticks']
        shots = self.shots

        # Unitary measurement ordering
        seq = [('Y90', 'Zero', 'Ym90','Ym90'),
               ('Y90', 'Zero', 'X90','X90'),
               ('Zero', 'Y90', 'Ym90', 'Ym90'),
               ('Zero', 'Y90', 'X90','X90'),
               ('Y90', 'X180', 'Ym90', 'Ym90'),
               ('Y90', 'X180', 'X90','X90')]

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
                for s in seq:
                    wait_for_trigger('q1')
                    wait_for_trigger('q2')
                    wait_for_trigger('c1')

                    align()
                    # reset_phase(q1)
                    # reset_phase(q2)
                    # reset_phase(c)
                    wait(reset_ticks, q1)
                    reset_frame(q1,q2,c)
                    align()
                    play(s[0], q1)
                    play(s[1], q2)
                    align()
                    self.circuit_macro(self.element_1, self.element_2, self.coupler, self.macro_args)
                    # you can add a circuit_macro dependent if loop
                    align()
                    play(s[2], q1)
                    play(s[3], q2)
                    align()
                    I1, Q1, I2, Q2 = measure_2q(self.element_1, self.element_2, I1, Q1, I2, Q2)

                    save(I1, I1_stream)#appends to the list
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

        counts_array = super().job_to_shots_2q(self.job, [self.shots, 6], self.calib_params) #self.shots,6
        counts_array = correct_counts_array(self.confusion,counts_array)


        raw_data = np.transpose(counts_array)
        # self.data = [counts_array, self.confusion]

        # self.data = raw_data

        data_holder = []

        for i in range(6):
            data_holder.append(
                {'counts': {'00': raw_data[i][0], '01': raw_data[i][1], '10': raw_data[i][2], '11': raw_data[i][3]}})

        shots = self.shots
#u22 and u11 should be flipped - done - if I wanna go back to previous one, u22 flips with u11, and u12 with u21, and u12e with u22e
        u22_xp = (raw_data[0][0] / shots - raw_data[0][1] / shots + raw_data[0][2] / shots - raw_data[0][3] / shots) # 00-01+10-11, measurin q2 in 00, left is q1, right is q2. Followed the notation from unitary_tompgraphy.py
        u22_yp = (raw_data[1][0] / shots - raw_data[1][1] / shots + raw_data[1][2] / shots - raw_data[1][3] / shots)  # coming from m_signle_shot_2Q.py
        u22 = u22_xp + 1j * u22_yp

        u21_xp = (raw_data[2][0] / shots - raw_data[2][1] / shots + raw_data[2][2] / shots - raw_data[2][3] / shots) # 00-01+10-11 #measuring q2 - q2 is on the right
        u21_yp = (raw_data[3][0] / shots - raw_data[3][1] / shots + raw_data[3][2] / shots - raw_data[3][3] / shots)
        u21 = u21_xp + 1j * u21_yp

        u11_xp = (raw_data[2][0] / shots + raw_data[2][1] / shots - raw_data[2][2] / shots - raw_data[2][3] / shots)# 00 + 01 - 10 - 11 # measuring qubit1
        u11_yp = (raw_data[3][0] / shots + raw_data[3][1] / shots - raw_data[3][2] / shots - raw_data[3][3] / shots)
        u11 = u11_xp + 1j * u11_yp

        u12_xp = (raw_data[0][0] / shots + raw_data[0][1] / shots - raw_data[0][2] / shots - raw_data[0][3] / shots) # 00+01 - 10 -11 # measuring qubit 1
        u12_yp = (raw_data[1][0] / shots + raw_data[1][1] / shots - raw_data[1][2] / shots - raw_data[1][3] / shots)
        u12 = u12_xp + 1j * u12_yp

        u22e_xp = (raw_data[4][0] / shots - raw_data[4][1] / shots + raw_data[4][2] / shots - raw_data[4][3] / shots) # 00 + 10  - 01 - 11 #measuring q2
        u22e_yp = (raw_data[5][0] / shots - raw_data[5][1] / shots + raw_data[5][2] / shots - raw_data[5][3] / shots)
        u22e = u22e_xp + 1j * u22e_yp

        u12e_xp = (raw_data[4][0] / shots + raw_data[4][1] / shots - raw_data[4][2] / shots - raw_data[4][3] / shots) # 00 + 01 - 10 -11 # meausring q1
        u12e_yp = (raw_data[5][0] / shots + raw_data[5][1] / shots - raw_data[5][2] / shots - raw_data[5][3] / shots)
        u12e = u12e_xp + 1j * u12e_yp

        if np.abs(u11) < np.abs(u21):
            u33 = u12e / np.conj(u21)
        else:
            u33 = u22e / np.conj(u11)

        # prob_dict = {'00':raw_data[i][0]/shots, '01':raw_data[i][1]/shots,'10':raw_data[i][2]/shots,'11':raw_data[i][3]/shots}

        # for r in tomo_init_data.analysis_results():
        #     print(r)

        print("u11 is ", u11, np.abs(u11))
        print("u12 is ", u12, np.abs(u12))
        print("u21 is ", u21, np.abs(u21))
        print("u22 is ", u22, np.abs(u22))
        print("u33 is ", u33, np.abs(u33))

        theta = np.arctan(np.abs(u12) / np.abs(u11))

        if np.abs(u21) > np.abs(u11):
            delta_p = np.angle(-u11 * u21)
        else:
            delta_p = np.angle(u11 * u22)

        if np.abs(u21) > np.abs(u11):
            phi = delta_p - np.angle(u12e * u21)
        else:
            phi = np.angle(u22) - np.angle(u22e)

        delta_m = 2 * np.angle(u11) - delta_p

        meta_dict = self.meta_dict

        psi_10 = - (meta_dict['q2']['q_LO']['LO_freq'] + meta_dict['q2']['q_IF'] - meta_dict['q1']['q_LO']['LO_freq'] -
                    meta_dict['q1']['q_IF']) * self.gate_len * 1e-9          # check if negative sign needs to be there - i think yes, because for us, wq1 > wq2. While in the paper it's opposite, detuning is positive
        # psi_10 = 0
        delta_moff = -2 * (np.angle(-u12 / 1j) + psi_10) + delta_p

        print("\ntheta is ", theta, np.abs(theta))
        print("phi is ", phi, np.abs(phi))
        print("delta_p is ", delta_p, np.abs(delta_p))
        print("delta_m is ", delta_m, np.abs(delta_m))
        print("delta_moff is ", delta_moff, np.abs(delta_moff))

        uni_tomo = {'mat_el': {'u11': u11, 'u12': u12, 'u21': u21, 'u22': u22, 'u33': u33},
                    'angles': {'theta': theta, 'phi': phi, 'delta_p': delta_p, 'delta_m': delta_m,
                               'delta_moff': delta_moff}

                    }


        self.data = [raw_data, uni_tomo, self.confusion]
















    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])