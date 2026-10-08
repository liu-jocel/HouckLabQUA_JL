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
from LabCode2.Helpers.gate_macros import flat_top_iSWAP
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qm import SimulationConfig
from LabCode2.Experiments.m_state_tomo import StateTomography

class GST_1Q(ExperimentClass):
    """
    XEB experiment
    """

    def __init__(self, element, num_ns, num_lengths, calib_params, outer_folder='', path='',
                 prefix='data', meta_dict=None, plot=True, fit=True, run=True, suffix=''):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict)

        self.element = element
        self.num_ns = num_ns
        self.num_lengths = num_lengths
        self.max_length = 2 ** self.num_lengths
        self.calib_params = calib_params
        self.plot = plot
        self.fit=fit
        self.outer_folder = outer_folder
        self.path = path
        self.prefix = prefix
        self.meta_dict = meta_dict
        self.germ_depths = [0, 1, 1, 2, 3]

        m_vec = np.logspace(1, self.num_lengths, self.num_lengths, base=2, dtype=int)
        self.m_vec = m_vec


        if run:
            self.job = self.acquire(self.make_prog())


    @staticmethod #just learned about decorators, this just makes it not take self as argument
    def play_fiducial(element, index):
        with switch_(index, unsafe=False):
            with case_(0):
                #(4, element)
                play("X90", element)
            with case_(1):
                play("X90", element)
            with case_(2):
                play("Y90", element)
            with case_(3):
                play("X90", element)
                play("X90", element)
            with case_(4):
                play("X90", element)
                play("X90", element)
            with case_(5):
                play("Y90", element)
                play("Y90", element)
                play("Y90", element)

    @staticmethod
    def play_germ(element, index, num_repeats, r):
        with switch_(index, unsafe=False):
            with case_(0):
                #wait(4, element)
                with for_(r, 0, r < num_repeats, r + 1):
                    play("X90", element)
            with case_(1):
                #wait(25, element)
                with for_(r, 0, r < num_repeats, r + 1):
                    play("X90", element)
            with case_(2):
                with for_(r, 0, r < num_repeats, r + 1):
                    play("X90", element)
            with case_(3):
                with for_(r, 0, r < num_repeats, r + 1):
                    play("Y90", element)
            with case_(4):
                with for_(r, 0, r < num_repeats, r + 1):
                    play("X90", element)
                    play("Y90", element)
            with case_(5):
                with for_(r, 0, r < num_repeats, r + 1):
                    play("X90", element)
                    play("X90", element)
                    play("Y90", element)

    @staticmethod
    def play_GST_seq(element,p,g,r,num_repeats,e):

        seq_vec = declare(int,value=[p]+[g+6]*num_repeats+[e])
        v = declare(int)

        with for_(v, 0, v < 2+num_repeats, v + 1):
            with switch_(seq_vec[v], unsafe=False):
                with case_(0):
                    (4, element)
                with case_(1):
                    play("X90", element)
                with case_(2):
                    play("Y90", element)
                with case_(3):
                    play("X90", element)
                    play("X90", element)
                with case_(4):
                    play("X90", element)
                    play("X90", element)
                with case_(5):
                    play("Y90", element)
                    play("Y90", element)
                    play("Y90", element)
                with case_(6):
                    # wait(4, element)
                    with for_(r, 0, r < num_repeats, r + 1):
                        play("X90", element)
                with case_(7):
                    # wait(25, element)
                    with for_(r, 0, r < num_repeats, r + 1):
                        play("X90", element)
                with case_(8):
                    with for_(r, 0, r < num_repeats, r + 1):
                        play("X90", element)
                with case_(9):
                    # with for_(r, 0, r < num_repeats, r + 1):
                    #     play("Y90", element)
                    play("Y90", element)
                with case_(10):
                    with for_(r, 0, r < num_repeats, r + 1):
                        play("X90", element)
                        play("Y90", element)
                with case_(11):
                    with for_(r, 0, r < num_repeats, r + 1):
                        play("X90", element)
                        play("X90", element)
                        play("Y90", element)

        # with switch_(p, unsafe=False):
        #     with case_(0):
        #         #(4, element)
        #         play("X90", element)
        #     with case_(1):
        #         play("X90", element)
        #     with case_(2):
        #         play("Y90", element)
        #     with case_(3):
        #         play("X90", element)
        #         play("X90", element)
        #     with case_(4):
        #         play("X90", element)
        #         play("X90", element)
        #     with case_(5):
        #         play("Y90", element)
        #         play("Y90", element)
        #         play("Y90", element)
        #
        # with switch_(g, unsafe=False):
        #     with case_(0):
        #         # wait(4, element)
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("X90", element)
        #     with case_(1):
        #         # wait(25, element)
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("X90", element)
        #     with case_(2):
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("X90", element)
        #     with case_(3):
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("Y90", element)
        #     with case_(4):
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("X90", element)
        #             play("Y90", element)
        #     with case_(5):
        #         with for_(r, 0, r < num_repeats, r + 1):
        #             play("X90", element)
        #             play("X90", element)
        #             play("Y90", element)
        #
        # with switch_(e, unsafe=False):
        #     with case_(0):
        #         #(4, element)
        #         play("X90", element)
        #     with case_(1):
        #         play("X90", element)
        #     with case_(2):
        #         play("Y90", element)
        #     with case_(3):
        #         play("X90", element)
        #         play("X90", element)
        #     with case_(4):
        #         play("X90", element)
        #         play("X90", element)
        #     with case_(5):
        #         play("Y90", element)
        #         play("Y90", element)
        #         play("Y90", element)



    def make_prog(self):

        m_vec = self.m_vec

        with program() as prog:
            n = declare(int) #indexes shots
            m = declare(int)
            p = declare(int) #indexes prep fiducials
            g = declare(int) #indexes germs
            e = declare(int) #indexes effect fiducials
            r = declare(int)  # indexes germ repeats
            m_vec_qua = declare(int, value=m_vec)
            germ_depths_qua = declare(int, value=self.germ_depths)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < self.num_ns, n + 1):
                with for_(m, 0, m < len(m_vec), m+1):
                    m_max = m_vec_qua[m]
                    with if_(germ_depths_qua[g] < m_max):
                        with for_(g, 0, g < 6, g+1):
                            with for_(e, 0, e < 6, e + 1):
                                with for_(p, 0, p < 6, p + 1):
                                    num_repeats = m_max / germ_depths_qua[g]
                                    self.play_GST_seq(self.element,p,g,r,num_repeats,e)
                                    align()

                                    measure_1q(self.meta_dict[self.element], I, Q)

                                    align()

                                    measure_1q(self.meta_dict[self.element], I, Q)

                                    save(I, I_stream)
                                    save(Q, Q_stream)

                # p = 3
                # g = 3
                # e = 3
                #
                # # p = declare(int,value=p)  # indexes prep fiducials
                # # g = declare(int,value=g)  # indexes germs
                # # e = declare(int,value=e)  # indexes effect fiducials
                #
                #
                # self.play_GST_seq(self.element,p,g,r,2000,e)
                # align()
                #
                # measure_1q(self.meta_dict[self.element], I, Q)
                #
                # save(I, I_stream)
                # save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")


        return prog


    def take_one_sequence(self,sequence,macro):

        tomo = StateTomography('q1', 'q2', 'c1', macro, sequence, self.num_ns, self.calib_params, meta_dict=self.meta_dict,
                                   outer_folder=self.outer_folder, path=self.path, confusion=self.confusion, plot=False)

        return tomo.data

    def take_XEB(self):

        data_dict = {}

        for k in range(self.num_ks):
            for m in self.m_vec:

                print(k,m)

                sequence = self.generate_sequence(m)

                data_ref = self.take_one_sequence(sequence, self.play_sequence_ref)
                data_int = self.take_one_sequence(sequence, self.play_sequence_int)

                data_dict[(m, k)] = {'sequence': sequence, 'ref': data_ref, 'int': data_int}

        self.data = data_dict



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])