from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.RBSupport import RBAnalysis
from LabCode2.Helpers.qua_helpers import measure_1q
import scipy as sp
import scipy.optimize
import random as rd
import math
from LabCode2.Helpers.gate_macros import flat_top_iSWAP
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qm import SimulationConfig
from LabCode2.Experiments.m_state_tomo import StateTomography
from LabCode2.Helpers.qua_helpers import measure_2q, measure_3q
from LabCode2.Helpers.SPAM_correction import correct_counts_array


def gauss_iSWAP(coupler):

    play('iSWAP_gauss', coupler)
    # align()

def fit_XEB(m_vec, data):
    """
    Fitting XEB Helper function
    """
    def fitRB(x, a, p, b):
        return a*p **(x) + b

    a_seed = data[0] + data[-1]
    b_seed = data[-1]
    p_seed = 0.99#1 - 0.01/a_seed

    try:
        popt, pcov = sp.optimize.curve_fit(fitRB, m_vec, data, [a_seed, p_seed, b_seed])#, bounds=([-1, 0, -1], [1, 1, 1]))
        xeb_dict = {
            "a": (popt[0], np.sqrt(pcov[0][0])),
            "p": (popt[1], np.sqrt(pcov[1][1])),
            "b": (popt[2], np.sqrt(pcov[2][2])),
            "fit": fitRB(m_vec, popt[0], popt[1], popt[2])
        }
    except:
        xeb_dict = {
            "a": None,
            "p": None,
            "b": None,
            "fit": None
        }
    return xeb_dict


class XEB(ExperimentClass):
    """
    XEB experiment
    """

    def __init__(self, element_1, element_2, element_c, coupler, num_ms, num_ks, num_ns, calib_params, outer_folder='', path='XEB',
                 suffix='data', meta_dict=None, plot=True, fit=True, run=True, confusion=np.eye(4), discrim=None, plot_POVMf = []):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict)

        self.plot_POVMf = plot_POVMf
        self.element_1 = element_1
        self.element_2 = element_2
        self.element_c = element_c
        self.coupler = coupler
        self.num_ms = num_ms
        self.num_ks = num_ks
        self.num_ns = num_ns
        self.max_m = 2 ** self.num_ms
        self.calib_params = calib_params
        self.confusion = confusion
        self.discrim = discrim
        self.plot = plot
        self.fit=fit

        m_vec = np.logspace(1, self.num_ms, self.num_ms, base=2, dtype=int)
        self.m_vec = m_vec
        self.num_ms = self.num_ms
        self.seed = rd.randint(1,1000000)

        if run:
            self.take_XEB()
            self.plot_XEB()
            self.pickle_data()

    def generate_sequence(self,m):

        sequence_list = np.zeros(m,dtype=int)

        for i in range(m):
            sequence_list[i] = np.random.randint(0,9)

        return sequence_list

    @staticmethod #just learned about decorators, this just makes it not take self as argument
    def play_sequence_ref(element_1, element_2, coupler, sequence_list):
        depth = declare(int,value=len(sequence_list))
        i = declare(int)
        sequence_list = declare(int,value=sequence_list)
        with for_(i, 0, i < depth, i + 1):
            with switch_(sequence_list[i], unsafe=False):
                with case_(0):
                    play("Id", element_1)
                    play("Id", element_2)
                with case_(1):
                    play("Id", element_1)
                    play("X90", element_2)
                with case_(2):
                    play("Id", element_1)
                    play("Y90", element_2)
                with case_(3):
                    play("X90", element_1)
                    play("Id", element_2)
                with case_(4):
                    play("X90", element_1)
                    play("X90", element_2)
                with case_(5):
                    play("X90", element_1)
                    play("Y90", element_2)
                with case_(6):
                    play("Y90", element_1)
                    play("Id", element_2)
                with case_(7):
                    play("Y90", element_1)
                    play("X90", element_2)
                with case_(8):
                    play("Y90", element_1)
                    play("Y90", element_2)

    @staticmethod  # just learned about decorators, this just makes it not take self as argument
    def play_sequence_int(element_1, element_2, coupler, sequence_list):
        depth = declare(int, value=len(sequence_list))
        i = declare(int)
        sequence_list = declare(int, value=sequence_list)
        with for_(i, 0, i < depth, i + 1):
            with switch_(sequence_list[i], unsafe=False):
                with case_(0):
                    play("Id", element_1)
                    play("Id", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(1):
                    play("Id", element_1)
                    play("X90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(2):
                    play("Id", element_1)
                    play("Y90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(3):
                    play("X90", element_1)
                    play("Id", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(4):
                    play("X90", element_1)
                    play("X90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(5):
                    play("X90", element_1)
                    play("Y90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(6):
                    play("Y90", element_1)
                    play("Id", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(7):
                    play("Y90", element_1)
                    play("X90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()
                with case_(8):
                    play("Y90", element_1)
                    play("Y90", element_2)
                    align()
                    flat_top_iSWAP(coupler)
                    # gauss_iSWAP(coupler)
                    align()

    def make_one_sequence(self,sequence,macro):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        coupler = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        num_ns = self.num_ns

        with program() as prog:
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            Ic = declare(fixed)
            Qc = declare(fixed)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()
            I2_stream = declare_stream()
            Q2_stream = declare_stream()
            Ic_stream = declare_stream()
            Qc_stream = declare_stream()

            with for_(n, 0, n < num_ns, n + 1):

                wait_for_trigger(qubit_1)
                wait(reset_ticks, qubit_1)

                macro(qubit_1, qubit_2, coupler, sequence)

                I1, Q1, I2, Q2, Ic, Qc = measure_3q(self.meta_dict[self.element_1], self.meta_dict[self.element_2],
                                                    self.meta_dict[self.element_c],
                                                    I1, Q1, I2, Q2, Ic, Qc)

                save(I1, I1_stream)
                save(Q1, Q1_stream)
                save(I2, I2_stream)
                save(Q2, Q2_stream)
                save(Ic, Ic_stream)#
                save(Qc, Qc_stream)#

            with stream_processing():
                I1_stream.save_all("I1")
                Q1_stream.save_all("Q1")
                I2_stream.save_all("I2")
                Q2_stream.save_all("Q2")
                Ic_stream.save_all("Ic")
                Qc_stream.save_all("Qc")

        return prog

    def make_simulation_prog(self):

        qubit_1 = self.meta_dict[self.element_1]['q_name']
        qubit_2 = self.meta_dict[self.element_2]['q_name']
        coupler = self.meta_dict[self.coupler]['c_name']
        reset_ticks = self.meta_dict['global_params']['reset_ticks']
        num_ns = self.num_ns

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

            with for_(n, 0, n < num_ns, n + 1):
                wait(reset_ticks, qubit_1)

                self.play_sequence_int('q1', 'q2', 'c1', [5,5,5,5])

                I1, Q1, I2, Q2 = measure_2q(self.meta_dict[self.element_1], self.meta_dict[self.element_2], I1, Q1,
                                            I2, Q2)

                save(I1, I1_stream)
                save(Q1, Q1_stream)
                save(I2, I2_stream)
                save(Q2, Q2_stream)

            with stream_processing():
                I1_stream.save_all("I_c1")
                Q1_stream.save_all("Q_c1")
                I2_stream.save_all("I_c2")
                Q2_stream.save_all("Q_c2")

        return prog


    def take_XEB(self):

        data_dict = {}

        for k in range(self.num_ks):
            for m in self.m_vec:

                print(k,m)

                sequence = self.generate_sequence(m)

                prog_ref = self.make_one_sequence(sequence,self.play_sequence_ref)
                job_ref = self.acquire(prog_ref)
                # data_ref = self.job_to_shots_2q(job_ref, [self.num_ns,1], self.calib_params)
                data_ref = self.job_to_shots_arb(job_ref, [self.num_ns,1], self.discrim)
                counts_array = np.zeros((len(data_ref), 1))
                for j, key in enumerate(self.discrim.POVM.keys()):
                    counts_array[j, :] = data_ref[key]
                data_ref = {k:correct_counts_array(self.confusion, counts_array)[i] for i,k in enumerate(data_ref.keys())}

                prog_int = self.make_one_sequence(sequence, self.play_sequence_int)
                job_int = self.acquire(prog_int)
                # data_int = self.job_to_shots_2q(job_int, [self.num_ns,1], self.calib_params)
                data_int = self.job_to_shots_arb(job_int, [self.num_ns, 1], self.discrim)
                counts_array = np.zeros((len(data_int), 1))
                for j, key in enumerate(self.discrim.POVM.keys()):
                    counts_array[j, :] = data_int[key]
                data_int = {k: correct_counts_array(self.confusion, counts_array)[i] for i, k in enumerate(data_int.keys())}


                data_dict[(m, k)] = {'sequence': sequence, 'ref': data_ref, 'int': data_int}

        self.data = data_dict



    def plot_XEB(self):

        data_m_ref = {}
        data_m_int = {}
        xeb_dict = {}

        for state in self.discrim.POVM.keys():
            p_vec_ref = np.zeros(len(self.m_vec))
            p_vec_int = np.zeros(len(self.m_vec))
            for i, m in enumerate(self.m_vec):
                for k in range(self.num_ks):
                    p_vec_ref[i] += self.data[(m, k)]['ref'][state] / self.num_ks
                    p_vec_int[i] += self.data[(m, k)]['int'][state] / self.num_ks

            data_m_ref[state] = p_vec_ref
            data_m_int[state] = p_vec_int

            try:

                xeb_dict[state] = {"ref": fit_XEB(self.m_vec, p_vec_ref),
                               "int": fit_XEB(self.m_vec, p_vec_int)
                               }
            except:
                pass
        if self.plot:
            colors = plt.cm.jet(np.linspace(0,1, len(self.discrim.POVM.keys())))
            fig, ax = plt.subplots(2, 1, figsize = (12,8), dpi = 120, constrained_layout=True)
            for c,state in enumerate(self.discrim.POVM.keys()):
                ax[0].plot(self.m_vec, data_m_ref[state], 'o', label=state, color = colors[c])
                ax[1].plot(self.m_vec, data_m_int[state], 'o', label=state, color = colors[c])

                for key in self.plot_POVMf:
                    if state == key:
                        try:
                            ax[0].plot(self.m_vec, xeb_dict[state]["ref"]["fit"], '-',
                                       label=f'{xeb_dict[state]["ref"]["p"][0]:.3f}\n+- {xeb_dict[state]["ref"]["p"][1]:.3f}', color = colors[c])
                            ax[1].plot(self.m_vec, xeb_dict[state]["int"]["fit"], '-',
                                       label=f'{xeb_dict[state]["int"]["p"][0]:.3f}\n+- {xeb_dict[state]["int"]["p"][1]:.3f}', color = colors[c])

                        except:
                            # raise TypeError('Could not fit')
                            pass

            ax[1].set_xlabel('Num Cycles')
            ax[0].set_ylabel('State probability')
            ax[1].set_ylabel('State probability')
            ax[0].legend(bbox_to_anchor=[1, 1], loc='upper left')
            ax[1].legend(bbox_to_anchor=[1, 1], loc='upper left')
            ax[0].set_title('Ref')
            ax[1].set_title('Int')

            # ax.set_title(f'RB on {q_name}, F = {self.gate_fidel:.4f}')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)


