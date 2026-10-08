from qm.qua import *
from qm import QuantumMachinesManager
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
from textwrap import wrap
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qm import SimulationConfig


class RB1Q(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, m_vec, num_ks, num_ns, calib_params, outer_folder='', path='', prefix='data',
                 meta_dict=None, plot=True, fit=True, interleaved=False):
        super().__init__(path=path, outer_folder=outer_folder, prefix=prefix, meta_dict=meta_dict)

        self.element = element
        self.num_ms = len(m_vec)
        self.num_ks = num_ks
        self.num_ns = num_ns
        self.max_m = 2 ** self.num_ms
        self.calib_params = calib_params
        self.plot = plot
        self.fit = fit
        self.data = {"data": {
            "meta_dict": meta_dict,
        },
        }

        # m_vec = np.logspace(1, self.num_ms, self.num_ms, base=2, dtype=int)
        self.m_vec_qua = [int(m_vec[i]) for i in range(int(self.num_ms))]
        self.m_vec = self.m_vec_qua
        self.num_ms = self.num_ms
        self.seed = rd.randint(1, 1000000)

    def make_prog(self, interleaved=False, simulate=False):

        element = self.element  # self.meta_dict[self.element]['q_name']

        # if simulate:
        #     reset_ticks = 20
        # else:
        reset_ticks = self.meta_dict['global_params']['reset_ticks']

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

        def generate_sequence_interleaved_x90():
            cayley = declare(int, value=c1_table.flatten().tolist())
            inv_list = declare(int, value=inv_gates)
            current_state = declare(int)
            step = declare(int)
            step_int = declare(int)
            sequence = declare(int, size=max_m + 1)
            inv_gate = declare(int, size=max_m + 1)
            i = declare(int)
            rand = Random(seed=seed)

            assign(current_state, 0)
            with for_(i, 0, i < max_m, i + 1):
                assign(step, rand.rand_int(24))
                assign(current_state, cayley[current_state * 24 + step])
                assign(sequence[i], step)
                assign(step_int, 12)
                assign(current_state, cayley[current_state * 24 + step_int])
                assign(inv_gate[i], inv_list[current_state])

            return sequence, inv_gate

        def play_sequence(sequence_list, depth):
            element_X90 = self.meta_dict[self.element]['q_element_X90']
            q_IF = self.meta_dict[self.element]['q_IF']
            i = declare(int)
            with for_(i, 0, i <= depth, i + 1):
                with switch_(sequence_list[i], unsafe=True):
                    with case_(0):
                        wait(25, element)
                    with case_(1):
                        play("X180", element)
                    with case_(2):
                        play("Y180", element)
                    with case_(3):
                        play("Y180", element)
                        play("X180", element)

                    with case_(4):
                        play("X90", element_X90)
                        play("Y90", element_X90)

                    with case_(5):
                        play("X90", element_X90)
                        play("Ym90", element_X90)

                    with case_(6):
                        play("Xm90", element_X90)
                        play("Y90", element_X90)

                    with case_(7):
                        play("Xm90", element_X90)
                        play("Ym90", element_X90)

                    with case_(8):
                        play("Y90", element_X90)
                        play("X90", element_X90)

                    with case_(9):
                        play("Y90", element_X90)
                        play("Xm90", element_X90)

                    with case_(10):
                        play("Ym90", element_X90)
                        play("X90", element_X90)

                    with case_(11):
                        play("Ym90", element_X90)
                        play("Xm90", element_X90)

                    with case_(12):
                        play("X90", element_X90)

                    with case_(13):
                        reset_frame()
                        play("Xm90", element_X90)

                    with case_(14):
                        reset_frame()
                        play("Y90", element_X90)

                    with case_(15):
                        reset_frame()
                        play("Ym90", element_X90)

                    with case_(16):
                        play("Xm90", element_X90)
                        play("Y90", element_X90)
                        play("X90", element_X90)

                    with case_(17):
                        play("Xm90", element_X90)
                        play("Ym90", element_X90)
                        play("X90", element_X90)

                    with case_(18):
                        play("X180", element)

                        play("Y90", element_X90)

                    with case_(19):
                        play("X180", element)

                        play("Ym90", element_X90)

                    with case_(20):
                        play("Y180", element)

                        play("X90", element_X90)

                    with case_(21):
                        play("Y180", element)

                        play("Xm90", element_X90)

                    with case_(22):
                        play("X90", element_X90)
                        play("Y90", element_X90)
                        play("X90", element_X90)

                    with case_(23):
                        play("Xm90", element_X90)
                        play("Y90", element_X90)
                        play("Xm90", element_X90)

        def play_sequence_XZ(sequence_list, depth):
            i = declare(int)
            with for_(i, 0, i <= depth, i + 1):
                with switch_(sequence_list[i], unsafe=True):
                    with case_(0):
                        wait(25, element)
                    with case_(1):
                        play("X180", element)
                    with case_(2):
                        play("X180", element)
                        frame_rotation_2pi(0.5, element)
                    with case_(3):
                        frame_rotation_2pi(0.5, element)
                    with case_(4):
                        frame_rotation_2pi(-0.25, element)
                        play("X90", element)
                    with case_(5):
                        # [cirq.S(cirq.LineQubit(0)), (cirq.X**0.5).on(cirq.LineQubit(0))]
                        frame_rotation_2pi(0.25, element)
                        play("X90", element)
                    with case_(6):
                        # [cirq.S(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))]
                        frame_rotation_2pi(0.25, element)
                        play("Xm90", element)
                    with case_(7):
                        # [(cirq.S**-1).on(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))],
                        frame_rotation_2pi(-0.25, element)
                        play("Xm90", element)
                    with case_(8):
                        # [(cirq.X**0.5).on(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                        play("X90", element)
                        frame_rotation_2pi(0.25, element)
                    with case_(9):
                        # [(cirq.X**-0.5).on(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))],
                        play("Xm90", element)
                        frame_rotation_2pi(-0.25, element)
                    with case_(10):
                        # [(cirq.X**0.5).on(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))],
                        play("X90", element)
                        frame_rotation_2pi(-0.25, element)
                    with case_(11):
                        # [(cirq.X**-0.5).on(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                        play("Xm90", element)
                        frame_rotation_2pi(0.25, element)
                    with case_(12):
                        play("X90", element)
                    with case_(13):
                        play("Xm90", element)
                    with case_(14):
                        # [(cirq.S ** -1).on(cirq.LineQubit(0)),
                        #  (cirq.X ** 0.5).on(cirq.LineQubit(0)),
                        #   cirq.S(cirq.LineQubit(0))],
                        play("Y90", element)
                    with case_(15):
                        # [(cirq.S ** -1).on(cirq.LineQubit(0)),
                        #  (cirq.X ** -0.5).on(cirq.LineQubit(0)),
                        #  cirq.S(cirq.LineQubit(0))],
                        play("Ym90", element)
                    with case_(16):
                        # [cirq.S(cirq.LineQubit(0))],
                        frame_rotation_2pi(0.25, element)
                    with case_(17):
                        # [(cirq.S**-1).on(cirq.LineQubit(0))]
                        frame_rotation_2pi(-0.25, element)
                    with case_(18):
                        # [(cirq.S ** -1).on(cirq.LineQubit(0)),
                        #  (cirq.X ** 0.5).on(cirq.LineQubit(0)),
                        #  (cirq.S ** -1).on(cirq.LineQubit(0))],
                        frame_rotation_2pi(-0.25, element)
                        play("X90", element)
                        frame_rotation_2pi(-0.25, element)
                    with case_(19):
                        # [cirq.S(cirq.LineQubit(0)),
                        #  (cirq.X ** 0.5).on(cirq.LineQubit(0)),
                        #  cirq.S(cirq.LineQubit(0))],
                        frame_rotation_2pi(0.25, element)
                        play("X90", element)
                        frame_rotation_2pi(0.25, element)
                    with case_(20):
                        # [cirq.Z(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))],
                        frame_rotation_2pi(0.5, element)
                        play("Xm90", element)
                    with case_(21):
                        # [cirq.Z(cirq.LineQubit(0)), (cirq.X**0.5).on(cirq.LineQubit(0))],
                        frame_rotation_2pi(0.5, element)
                        play("X90", element)
                    with case_(22):
                        # [cirq.X(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                        play("X180", element)
                        frame_rotation_2pi(0.25, element)
                    with case_(23):
                        # [cirq.X(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))]]
                        play("X180", element)
                        frame_rotation_2pi(-0.25, element)

        def play_sequence_interleaved_x90(sequence_list, depth):
            i = declare(int)
            with for_(i, 0, i <= depth, i + 1):
                with switch_(sequence_list[i], unsafe=True):
                    with case_(0):
                        wait(25, element)
                        play("X90", element)
                    with case_(1):
                        play("X180", element)
                        play("X90", element)
                    with case_(2):
                        play("Y180", element)
                        play("X90", element)
                    with case_(3):
                        play("Y180", element)
                        play("X180", element)
                        play("X90", element)
                    with case_(4):
                        play("X90", element)
                        play("Y90", element)
                        play("X90", element)
                    with case_(5):
                        play("X90", element)
                        play("Ym90", element)
                        play("X90", element)
                    with case_(6):
                        play("Xm90", element)
                        play("Y90", element)
                        play("X90", element)
                    with case_(7):
                        play("Xm90", element)
                        play("Ym90", element)
                        play("X90", element)
                    with case_(8):
                        play("Y90", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(9):
                        play("Y90", element)
                        play("Xm90", element)
                        play("X90", element)
                    with case_(10):
                        play("Ym90", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(11):
                        play("Ym90", element)
                        play("Xm90", element)
                        play("X90", element)
                    with case_(12):
                        play("X90", element)
                        play("X90", element)
                    with case_(13):
                        play("Xm90", element)
                        play("X90", element)
                    with case_(14):
                        play("Y90", element)
                        play("X90", element)
                    with case_(15):
                        play("Ym90", element)
                        play("X90", element)
                    with case_(16):
                        play("Xm90", element)
                        play("Y90", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(17):
                        play("Xm90", element)
                        play("Ym90", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(18):
                        play("X180", element)
                        play("Y90", element)
                        play("X90", element)
                    with case_(19):
                        play("X180", element)
                        play("Ym90", element)
                        play("X90", element)
                    with case_(20):
                        play("Y180", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(21):
                        play("Y180", element)
                        play("Xm90", element)
                        play("X90", element)
                    with case_(22):
                        play("X90", element)
                        play("Y90", element)
                        play("X90", element)
                        play("X90", element)
                    with case_(23):
                        play("Xm90", element)
                        play("Y90", element)
                        play("Xm90", element)
                        play("X90", element)

        with program() as prog:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            s = declare(int)
            ms = declare(int, value=m_vec_qua)
            saved_gate = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            seq_stream = declare_stream()

            with for_(k, 0, k < num_ks, k + 1):
                sequence_list, inv_gate_list = generate_sequence()

                with for_(s, 0, s < max_m, s + 1):
                    save(sequence_list[s], seq_stream)

                with for_(m, 0, m < num_ms, m + 1):
                    m_pow = ms[m]
                    # m_pow = m_pow * 2
                    with for_(n, 0, n < num_ns, n + 1):
                        assign(saved_gate, sequence_list[m_pow])
                        assign(sequence_list[m_pow], inv_gate_list[m_pow - 1])

                        wait(reset_ticks, element)
                        align()

                        play_sequence(sequence_list, m_pow)

                        align()
                        I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                        save(I, I_stream)
                        save(Q, Q_stream)

                        assign(sequence_list[m_pow], saved_gate)
            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
                seq_stream.save_all("circuits")

        with program() as prog_int:
            k = declare(int)
            m = declare(int)
            n = declare(int)
            s = declare(int)
            ms = declare(int, value=m_vec_qua)
            saved_gate = declare(int)
            inv_gate_to_play = declare(int, value=[0])
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            seq_stream = declare_stream()

            with for_(k, 0, k < num_ks, k + 1):
                sequence_list, inv_gate_list = generate_sequence_interleaved_x90()

                with for_(s, 0, s < max_m, s + 1):
                    save(sequence_list[s], seq_stream)

                with for_(m, 0, m < num_ms, m + 1):
                    m_pow = ms[m]
                    with for_(n, 0, n < num_ns, n + 1):
                        assign(saved_gate, sequence_list[m_pow])
                        assign(sequence_list[m_pow], inv_gate_list[m_pow - 1])
                        assign(inv_gate_to_play[0], inv_gate_list[m_pow - 1])

                        wait(reset_ticks, element)
                        align()

                        play_sequence_interleaved_x90(sequence_list, m_pow - 1)
                        play_sequence(inv_gate_to_play, 0)

                        align()
                        I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                        save(I, I_stream)
                        save(Q, Q_stream)

                        assign(sequence_list[m_pow], saved_gate)
            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
                seq_stream.save_all("circuits")

        if interleaved:
            return prog_int
        else:
            return prog

    def acquire_RB(self, interleaved):

        self.job = self.acquire(self.make_prog(interleaved))

        res_handles = self.job.result_handles
        I_ss_handle = res_handles.get("I")
        Q_ss_handle = res_handles.get("Q")
        circuit_handle = res_handles.get("circuits")

        I_ss_handle.wait_for_all_values()
        Q_ss_handle.wait_for_all_values()
        circuit_handle.wait_for_all_values()
        I_ss_array = I_ss_handle.fetch_all()['value']
        Q_ss_array = Q_ss_handle.fetch_all()['value']
        self.circuits = circuit_handle.fetch_all()['value']
        print("QUA program results have been returned")

        results = [I_ss_array, Q_ss_array]
        data = {'data': {'results': results, 'xpts': self.m_vec}}
        self.data = data

        return data

    def analyze(self):

        ss_array = super().job_to_shots_1q(self.job, [self.num_ks, self.num_ms, self.num_ns], self.calib_params)

        self.data["data"] = {"m": self.m_vec,
                             "single shot": ss_array,
                             }

        def fitRB(x, a, p, b):
            return a * p ** x + b

        self.gate_fidel = 0

        fit_array = ss_array.mean(axis=(0, 2))

        return_prob = 1 - fit_array

        if self.fit:
            popt, pcov = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob,
                                               [0.3, 0.99, 0.5], maxfev=5000)

            a = popt[0]
            p = popt[1]
            p_err = np.sqrt(pcov[1][1])
            b = popt[2]
            self.data['data']["fit"] = popt, pcov
            self.clif_fidel = 1 - 0.5 * (1 - p)
            self.gate_fidel = 1 - 0.5 * (1 - p) / 1.875
            self.gate_fidel_err = p_err * 0.5 / 1.875
            self.amp = a
            self.off = b

            print(self.gate_fidel)

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.m_vec, return_prob, 'o')
            if self.fit:
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                        fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt))
            ax.set_xlabel('Num Cliffords')
            ax.set_ylabel('Return probability')
            q_name = self.element
            ax.set_title(f'RB on {q_name} one freq, F = {self.gate_fidel:.4f}')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)

        return ss_array

    def analyze_leak(self):

        ss_array = super().job_to_shots_1q_3state(self.job, [self.num_ks, self.num_ms, self.num_ns], self.calib_params)

        def fitRB(x, a, p, b):
            return a * p ** x + b

        self.gate_fidel = 0

        probs = np.zeros([3, self.num_ms])

        self.data = [self.m_vec, ss_array, probs]

        num_shots_per_m = self.num_ks * self.num_ns

        for k in range(self.num_ks):
            for n in range(self.num_ns):
                for m in range(self.num_ms):
                    probs[ss_array[k][m][n]][m] += 1 / num_shots_per_m

        if self.fit:

            try:

                popt, pcov = sp.optimize.curve_fit(fitRB, self.m_vec, probs[0],
                                                   [0.5, 0.99, 0.5])

                popt_leak, pcov_leak = sp.optimize.curve_fit(fitRB, self.m_vec, probs[2],
                                                             [-0.1, 0.99, 0.1])

            except:

                popt = [0, 1, 0]
                pcov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

                popt_leak = [0, 1, 0]
                pcov_leak = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            a = popt[0]
            p = popt[1]
            p_err = np.sqrt(pcov[1][1])
            b = popt[2]

            self.clif_fidel = 1 - 0.5 * (1 - p)
            self.gate_fidel = 1 - 0.5 * (1 - p) / 1.875
            self.gate_fidel_err = p_err * 0.5 / 1.875
            self.amp = a
            self.off = b
            self.leak_rate = (1 - popt_leak[1]) * popt_leak[2] / 1.875
            self.leak_rate_err = np.sqrt(pcov_leak[1][1]) * popt_leak[2] / 1.875

            print(self.gate_fidel)
            print(self.leak_rate)

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(self.m_vec, probs[0], 'o')
            ax.plot(self.m_vec, probs[1], 'o')
            ax.plot(self.m_vec, probs[2] * 20, 'o')
            if self.fit:
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                        fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt))
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                        20 * fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_leak))
            ax.set_xlabel('Num Cliffords')
            ax.set_ylabel('Return probability')
            q_name = self.element
            ax.set_title(f'RB on {q_name}, F = {self.gate_fidel:.3f}')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)

        return ss_array

    def display_ss(self, calib_params=None, data=None, plotDisp=False, figNum=1, **kwargs):
        if data is None:
            data = self.data

        if calib_params is None:
            raise Exception("No calibration parameters")

        ss_thresh = calib_params[0]
        ss_factor = calib_params[1]
        ss_theta = calib_params[2]

        m_vec = (data['data']['xpts'])
        I_array = data['data']['results'][0]
        Q_array = data['data']['results'][1]

        ss_array = ss_factor * np.real(np.exp(-1j * ss_theta) * (I_array + 1j * Q_array))

        fit_array = np.zeros(self.num_ms)
        fit_array_export = np.zeros([self.num_ms, self.num_ks, self.num_ns])

        for k in range(self.num_ks):
            for m in range(self.num_ms):
                for n in range(self.num_ns):
                    shot = ss_array[k * self.num_ms * self.num_ns + m * self.num_ns + n]
                    if shot > ss_thresh:
                        fit_array[m] = fit_array[m] + 1 / (self.num_ks * self.num_ns)
                        fit_array_export[m][k][n] = 1

        self.data_to_export = fit_array_export

        def fitRB(x, a, p, b):
            return a * p ** x + b

        return_prob = 1 - fit_array

        popt, pcov = sp.optimize.curve_fit(fitRB, m_vec, return_prob,
                                           [0.3, 0.99, 0.5])

        a = popt[0]
        p = popt[1]
        p_err = np.sqrt(pcov[1][1])
        b = popt[2]

        self.clif_fidel = 1 - 0.5 * (1 - p)
        self.gate_fidel = 1 - 0.5 * (1 - p) / 1.875
        self.gate_fidel_err = p_err * 0.5 / 1.875
        self.amp = a
        self.off = b

        print(self.gate_fidel)

        if self.plot:
            while plt.fignum_exists(num=figNum):
                figNum += 1
            fig, ax = plt.subplots(1, 1, num=figNum)

            plt.plot(m_vec, return_prob, 'o')
            plt.plot(np.linspace(m_vec[0], m_vec[-1]), fitRB(np.linspace(m_vec[0], m_vec[-1]), *popt))
            plt.xlabel('Num Cliffords')
            plt.ylabel('Return probability')
            plt.title(f'RB on Q2, F = {self.gate_fidel:.3f}')
            plt.savefig(self.iname, bbox_inches="tight")

            if plotDisp:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                fig.clf(True)
                plt.close(fig)

        return self.data_to_export

    def fidel_vs_k(self, calib_params, figNum=1, plotDisp=False):

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

        fidels = np.zeros([num_ks, num_ms])

        for k in range(num_ks):
            for m in range(num_ms):
                for n in range(num_ns):
                    shot = ss_array[k * num_ms * num_ns + m * num_ns + n]
                    if shot > ss_thresh:
                        fidels[k][m] = fidels[k][m] + 1 / (num_ns)

        if self.plot:
            while plt.fignum_exists(num=figNum):
                figNum += 1
            fig, ax = plt.subplots(1, 1, num=figNum)
            return_prob = 1 - fidels
            plt.plot(m_vec, return_prob.T, 'o')
            plt.xlabel('Num Cliffords')
            plt.ylabel('Return probability')
            plt.title('RB on Q2')
            plt.savefig(self.iname[:-4] + '_allk.png', bbox_inches="tight")

            if plotDisp:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                fig.clf(True)
                plt.close(fig)

        self.fidels = fidels

    def run_RB(self):

        self.acquire_RB(interleaved=False)
        ref_data = self.analyze()
        # RBAnalysis('Int', 1, ['ref', 'X90'], self.m_vec, ref_data)

    def run_interleaved_RB(self, plotDisp=False, figNum=1):

        self.acquire_RB(interleaved=False)
        ref_data = self.analyze()
        self.acquire_RB(interleaved=True)
        int_data = self.analyze()

        full_data = np.zeros([2, self.num_ks, self.num_ms, self.num_ns])
        full_data[0] = ref_data
        full_data[1] = int_data

        analysis = RBAnalysis('Int', 1, ['ref', 'X90'], self.m_vec, full_data)

        # analysis.runBootstrap()
        # analysis.printResults(analysis.bootResults)

        def fitRB(x, a, p, b):
            return a * p ** x + b

        return_prob_ref = 1 - analysis.traces[0]
        return_prob_int = 1 - analysis.traces[1]

        popt_ref, pcov_ref = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_ref,
                                                   [0.3, 0.99, 0.5])

        popt_int, pcov_int = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_int,
                                                   [0.3, 0.99, 0.5])

        while plt.fignum_exists(num=figNum):
            figNum += 1
        fig, ax = plt.subplots(1, 1, num=figNum)

        plt.plot(self.m_vec, return_prob_ref, 'bo', label='Ref')
        plt.plot(self.m_vec, return_prob_int, 'ro', label='X90')
        plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                 fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_ref), 'b', label=None)
        plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                 fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_int), 'r', label=None)
        plt.legend()
        q_name = self.element
        plt.xlabel('Num Cliffords')
        plt.ylabel('Return probability')
        plt.title(f'IRB on {q_name}, F = {analysis.naiveResults[0][0]:.3f}+/-{analysis.naiveResults[0][1]:.3f}')
        plt.savefig(self.iname, bbox_inches="tight")

        if plotDisp:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            fig.clf(True)
            plt.close(fig)

        file_name = self.iname[:-4] + '_IRB_data.pkl'
        output = dict(data=self.data, analysis=analysis)
        with open(file_name, 'wb') as f:
            pickle.dump(output, f)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])