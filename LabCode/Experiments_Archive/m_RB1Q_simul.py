from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
from LabCode.Helpers.config import gen_config
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.RBSupport import RBAnalysis
from LabCode2.Helpers.qua_helpers import measure_1q
import scipy as sp
import scipy.optimize
import random as rd
import math
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qm import SimulationConfig

class RB1QSimul(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element_1, element_2, num_ms, num_ks, num_ns, calib_params_1, calib_params_2, outer_folder='', path='', prefix='data', meta_dict=None, plot=True, fit=True):
        super().__init__(path=path, outer_folder=outer_folder,prefix=prefix, meta_dict=meta_dict)

        self.element_1 = element_1
        self.element_2 = element_2
        self.num_ms = num_ms
        self.num_ks = num_ks
        self.num_ns = num_ns
        self.max_m = 2 ** self.num_ms
        self.calib_params_1 = calib_params_1
        self.calib_params_2 = calib_params_2
        self.plot = plot
        self.fit=fit

        m_vec = np.logspace(1, self.num_ms, self.num_ms, base=2, dtype=int)
        self.m_vec_qua = [int(m_vec[i]) for i in range(int(self.num_ms))]
        self.m_vec = self.m_vec_qua
        self.num_ms = self.num_ms
        self.seed = rd.randint(1,1000000)


    def make_prog(self,interleaved=False):

        element_1 = self.meta_dict[self.element_1]['q_name']
        element_2 = self.meta_dict[self.element_2]['q_name']
        reset_ticks = self.meta_dict[self.element_1]['reset_ticks']

        m_vec_qua = self.m_vec_qua
        num_ms = self.num_ms
        num_ks = self.num_ks
        num_ns = self.num_ns
        max_m = self.max_m
        seed = self.seed

        inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

        def generate_sequence():
            cayley = declare(int, value=c1_table.flatten().tolist())
            inv_list = declare(int, value=inv_gates)
            current_state = declare(int)
            step = declare(int)
            sequence = declare(int, size=max_m + 1)
            inv_gate = declare(int, size=max_m + 1)
            i = declare(int)
            rand = Random(seed=seed)

            assign(current_state, 0)
            with for_(i, 0, i < max_m, i + 1):
                assign(step, rand.rand_int(24))
                assign(current_state, cayley[current_state * 24 + step])
                assign(sequence[i], step)
                assign(inv_gate[i], inv_list[current_state])

            return sequence, inv_gate

        def play_sequence(sequence_list_1, depth):
            i = declare(int)
            with for_(i, 0, i <= depth, i + 1):
                with switch_(sequence_list_1[i], unsafe=True):
                    with case_(0):
                        wait(25, element_1)
                    with case_(1):
                        play("X180", element_1)
                        play("X180", element_2)
                    with case_(2):
                        play("Y180", element_1)
                        play("Y180", element_2)
                    with case_(3):
                        play("Y180", element_1)
                        play("X180", element_1)
                        play("Y180", element_2)
                        play("X180", element_2)
                    with case_(4):
                        play("X90", element_1)
                        play("Y90", element_1)
                        play("X90", element_2)
                        play("Y90", element_2)
                    with case_(5):
                        play("X90", element_1)
                        play("Ym90", element_1)
                        play("X90", element_2)
                        play("Ym90", element_2)
                    with case_(6):
                        play("Xm90", element_1)
                        play("Y90", element_1)
                        play("Xm90", element_2)
                        play("Y90", element_2)
                    with case_(7):
                        play("Xm90", element_1)
                        play("Ym90", element_1)
                        play("Xm90", element_2)
                        play("Ym90", element_2)
                    with case_(8):
                        play("Y90", element_1)
                        play("X90", element_1)
                        play("Y90", element_2)
                        play("X90", element_2)
                    with case_(9):
                        play("Y90", element_1)
                        play("Xm90", element_1)
                        play("Y90", element_2)
                        play("Xm90", element_2)
                    with case_(10):
                        play("Ym90", element_1)
                        play("X90", element_1)
                        play("Ym90", element_2)
                        play("X90", element_2)
                    with case_(11):
                        play("Ym90", element_1)
                        play("Xm90", element_1)
                        play("Ym90", element_2)
                        play("Xm90", element_2)
                    with case_(12):
                        play("X90", element_1)
                        play("X90", element_2)
                    with case_(13):
                        play("Xm90", element_1)
                        play("Xm90", element_2)
                    with case_(14):
                        play("Y90", element_1)
                        play("Y90", element_2)
                    with case_(15):
                        play("Ym90", element_1)
                        play("Ym90", element_2)
                    with case_(16):
                        play("Xm90", element_1)
                        play("Y90", element_1)
                        play("X90", element_1)
                        play("Xm90", element_2)
                        play("Y90", element_2)
                        play("X90", element_2)
                    with case_(17):
                        play("Xm90", element_1)
                        play("Ym90", element_1)
                        play("X90", element_1)
                        play("Xm90", element_2)
                        play("Ym90", element_2)
                        play("X90", element_2)
                    with case_(18):
                        play("X180", element_1)
                        play("Y90", element_1)
                        play("X180", element_2)
                        play("Y90", element_2)
                    with case_(19):
                        play("X180", element_1)
                        play("Ym90", element_1)
                        play("X180", element_2)
                        play("Ym90", element_2)
                    with case_(20):
                        play("Y180", element_1)
                        play("X90", element_1)
                        play("Y180", element_2)
                        play("X90", element_2)
                    with case_(21):
                        play("Y180", element_1)
                        play("Xm90", element_1)
                        play("Y180", element_2)
                        play("Xm90", element_2)
                    with case_(22):
                        play("X90", element_1)
                        play("Y90", element_1)
                        play("X90", element_1)
                        play("X90", element_2)
                        play("Y90", element_2)
                        play("X90", element_2)
                    with case_(23):
                        play("Xm90", element_1)
                        play("Y90", element_1)
                        play("Xm90", element_1)
                        play("Xm90", element_2)
                        play("Y90", element_2)
                        play("Xm90", element_2)

                # with switch_(sequence_list_2[i], unsafe=True):
                #     with case_(0):
                #         wait(25, element_2)
                #     with case_(1):
                #         play("X180", element_2)
                #     with case_(2):
                #         play("Y180", element_2)
                #     with case_(3):
                #         play("Y180", element_2)
                #         play("X180", element_2)
                #     with case_(4):
                #         play("X90", element_2)
                #         play("Y90", element_2)
                #     with case_(5):
                #         play("X90", element_2)
                #         play("Ym90", element_2)
                #     with case_(6):
                #         play("Xm90", element_2)
                #         play("Y90", element_2)
                #     with case_(7):
                #         play("Xm90", element_2)
                #         play("Ym90", element_2)
                #     with case_(8):
                #         play("Y90", element_2)
                #         play("X90", element_2)
                #     with case_(9):
                #         play("Y90", element_2)
                #         play("Xm90", element_2)
                #     with case_(10):
                #         play("Ym90", element_2)
                #         play("X90", element_2)
                #     with case_(11):
                #         play("Ym90", element_2)
                #         play("Xm90", element_2)
                #     with case_(12):
                #         play("X90", element_2)
                #     with case_(13):
                #         play("Xm90", element_2)
                #     with case_(14):
                #         play("Y90", element_2)
                #     with case_(15):
                #         play("Ym90", element_2)
                #     with case_(16):
                #         play("Xm90", element_2)
                #         play("Y90", element_2)
                #         play("X90", element_2)
                #     with case_(17):
                #         play("Xm90", element_2)
                #         play("Ym90", element_2)
                #         play("X90", element_2)
                #     with case_(18):
                #         play("X180", element_2)
                #         play("Y90", element_2)
                #     with case_(19):
                #         play("X180", element_2)
                #         play("Ym90", element_2)
                #     with case_(20):
                #         play("Y180", element_2)
                #         play("X90", element_2)
                #     with case_(21):
                #         play("Y180", element_2)
                #         play("Xm90", element_2)
                #     with case_(22):
                #         play("X90", element_2)
                #         play("Y90", element_2)
                #         play("X90", element_2)
                #     with case_(23):
                #         play("Xm90", element_2)
                #         play("Y90", element_2)
                #         play("Xm90", element_2)

        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            s = declare(int)
            ms = declare(int, value=m_vec_qua)
            saved_gate_1 = declare(int)
            # saved_gate_2 = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()
            seq_stream = declare_stream()

            with for_(k, 0, k < num_ks, k + 1):
                sequence_list_1, inv_gate_list_1 = generate_sequence()
                # sequence_list_2, inv_gate_list_2 = generate_sequence()

                # with for_(s, 0, s < max_m, s + 1):
                #     save(sequence_list[s], seq_stream)

                with for_(m, 0, m < num_ms, m + 1):
                    m_pow = ms[m]
                    # m_pow = m_pow * 2
                    with for_(n, 0, n < num_ns, n + 1):
                        assign(saved_gate_1, sequence_list_1[m_pow])
                        assign(sequence_list_1[m_pow], inv_gate_list_1[m_pow - 1])
                        # assign(saved_gate_2, sequence_list_2[m_pow])
                        # assign(sequence_list_2[m_pow], inv_gate_list_2[m_pow - 1])

                        wait(reset_ticks, element_1)
                        align()

                        play_sequence(sequence_list_1, m_pow)

                        align()
                        I1, Q1 = measure_1q(self.meta_dict[self.element_1], I1, Q1)
                        I2, Q2 = measure_1q(self.meta_dict[self.element_2], I2, Q2)

                        save(I1, I1_stream)
                        save(Q1, Q1_stream)
                        save(I2, I2_stream)
                        save(Q2, Q2_stream)

                        assign(sequence_list_1[m_pow], saved_gate_1)
                        # assign(sequence_list_2[m_pow], saved_gate_2)
            with stream_processing():
                I1_stream.save_all("I1")
                Q1_stream.save_all("Q1")
                I2_stream.save_all("I2")
                Q2_stream.save_all("Q2")
                # seq_stream.save_all("circuits")

        return prog


    def acquire_RB(self):

        self.job = super().acquire(self.make_prog())

        res_handles = self.job.result_handles
        I1_ss_handle = res_handles.get("I1")
        Q1_ss_handle = res_handles.get("Q1")
        I2_ss_handle = res_handles.get("I2")
        Q2_ss_handle = res_handles.get("Q2")
        # circuit_handle = res_handles.get("circuits")

        I1_ss_handle.wait_for_all_values()
        Q1_ss_handle.wait_for_all_values()
        I2_ss_handle.wait_for_all_values()
        Q2_ss_handle.wait_for_all_values()
        # circuit_handle.wait_for_all_values()
        I1_ss_array = I1_ss_handle.fetch_all()['value']
        Q1_ss_array = Q1_ss_handle.fetch_all()['value']
        I2_ss_array = I2_ss_handle.fetch_all()['value']
        Q2_ss_array = Q2_ss_handle.fetch_all()['value']
        # self.circuits = circuit_handle.fetch_all()['value']
        print("QUA program results have been returned")

        results=[I1_ss_array, Q1_ss_array, I2_ss_array, Q2_ss_array]
        data={'data': {'results': results, 'xpts':self.m_vec}}
        self.data=data

        return data


    def simulate(self, channels, ticks, ticks_to_plot):

        self.samples = super().simulate(self.make_prog(), channels, ticks, ticks_to_plot)

    def analyze(self):

        ss_array_1 = super().job_to_shots_1q(self.job, [self.num_ks, self.num_ms, self.num_ns], self.calib_params_1, I_name='I1', Q_name='Q1')
        ss_array_2 = super().job_to_shots_1q(self.job, [self.num_ks, self.num_ms, self.num_ns], self.calib_params_2, I_name='I2', Q_name='Q2')
        self.data = [self.m_vec, ss_array_1, ss_array_2]

        def fitRB(x,a,p,b):
            return a*p**x+b

        fit_array_1 = ss_array_1.mean(axis=(0,2))
        fit_array_2 = ss_array_2.mean(axis=(0, 2))

        return_prob_1 = 1 - fit_array_1
        return_prob_2 = 1 - fit_array_2

        self.gate_fidel_1 = 0
        self.gate_fidel_2 = 0

        if self.fit:
            popt_1, pcov_1 = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_1,
                                               [0.3, 0.99, 0.5])
            popt_2, pcov_2 = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_2,
                                                   [0.3, 0.99, 0.5])

            a1 = popt_1[0]
            p1 = popt_1[1]
            p1_err = np.sqrt(pcov_1[1][1])
            b1 = popt_1[2]

            a2 = popt_2[0]
            p2 = popt_2[1]
            p2_err = np.sqrt(pcov_2[1][1])
            b2 = popt_2[2]

            self.clif_fidel = 1 - 0.5 * (1 - p1)
            self.gate_fidel_1 = 1 - 0.5 * (1 - p1) / 1.875
            self.gate_fidel_2 = 1 - 0.5 * (1 - p2) / 1.875
            self.gate_fidel_1_err = p1_err * 0.5/1.875
            self.gate_fidel_2_err = p2_err * 0.5 / 1.875
            self.amp = a1
            self.off = b1

            print(self.gate_fidel_1)

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.m_vec, return_prob_1, 'o')
            ax.plot(self.m_vec, return_prob_2, 'o')
            if self.fit:
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]), fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_1))
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                        fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_2))
            ax.set_xlabel('Num Cliffords')
            ax.set_ylabel('Return probability')
            ax.set_title(f'Simultaneous RB, F = {self.gate_fidel_1:.3f},{self.gate_fidel_2:.3f}')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)

        return ss_array_1

    def fidel_vs_k(self, calib_params,figNum=1,plotDisp=False):

        data = self.data
        num_ks = self.num_ks
        num_ms = self.num_ms
        num_ns = self.num_ns
        m_vec = (data['data']['xpts'])

        ss_thresh = calib_params[0]
        ss_factor = calib_params[1]
        ss_theta = calib_params[2]

        I_array = data['data']['results'][0]
        Q_array = data['data']['results'][1]

        ss_array = ss_factor * np.real(np.exp(-1j * ss_theta) * (I_array + 1j * Q_array))

        fidels = np.zeros([num_ks,num_ms])

        for k in range(num_ks):
            for m in range(num_ms):
                for n in range(num_ns):
                    shot = ss_array[k*num_ms*num_ns+m*num_ns+n]
                    if shot>ss_thresh:
                        fidels[k][m] = fidels[k][m] + 1/(num_ns)

        if self.plot:
            while plt.fignum_exists(num=figNum):
                figNum +=1
            fig,ax = plt.subplots(1,1,num=figNum)
            return_prob = 1-fidels
            plt.plot(m_vec, return_prob.T, 'o')
            plt.xlabel('Num Cliffords')
            plt.ylabel('Return probability')
            plt.title('RB on Q2')
            plt.savefig(self.iname[:-4]+'_allk.png', bbox_inches="tight")

            if plotDisp:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                fig.clf(True)
                plt.close(fig)


        self.fidels = fidels

    def run_interleaved_RB(self, plotDisp=False,figNum=1):

        self.acquire(interleaved=False)
        ref_data = self.analyze()
        self.acquire(interleaved=True)
        int_data = self.analyze()

        full_data = np.zeros([2,self.num_ks,self.num_ms,self.num_ns])
        full_data[0] = ref_data
        full_data[1] = int_data

        analysis = RBAnalysis('Int',1,['ref','X90'],self.m_vec,full_data)
        analysis.runBootstrap()
        analysis.printResults(analysis.bootResults)

        def fitRB(x,a,p,b):
            return a*p**x+b

        return_prob_ref = 1 - analysis.traces[0]
        return_prob_int = 1 - analysis.traces[1]

        popt_ref, pcov_ref = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_ref,
                                             [0.3,0.99,0.5])

        popt_int, pcov_int = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_int,
                                           [0.3, 0.99, 0.5])

        while plt.fignum_exists(num=figNum):
            figNum += 1
        fig, ax = plt.subplots(1, 1, num=figNum)

        plt.plot(self.m_vec, return_prob_ref, 'bo',label='Ref')
        plt.plot(self.m_vec, return_prob_int, 'ro',label='X90')
        plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]), fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_ref),'b',label=None)
        plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]), fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_int),'r',label=None)
        plt.legend()
        q_name = self.element['q_name']
        plt.xlabel('Num Cliffords')
        plt.ylabel('Return probability')
        plt.title(f'IRB on {q_name}, F = {analysis.bootResults[0][0]:.3f}+/-{analysis.bootResults[0][1]:.3f}')
        plt.savefig(self.iname, bbox_inches="tight")

        if plotDisp:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            fig.clf(True)
            plt.close(fig)

        file_name = self.iname[:-4]+'_IRB_data.pkl'
        with open(file_name,'wb') as f:
            pickle.dump(analysis,f)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])