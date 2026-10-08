from qm.qua import *
from qm import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Helpers.qua_helpers import measure_1q
from LabCode2.Config.config_helpers import Omega_G, Omega_G_dot
import scipy as sp
import scipy.optimize
import random as rd
import time
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.bakery.bakery import baking
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q, meta_dict_to_config_6Q
from LabCode2.Helpers.gate_macros import phased_XZ
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q

class RB1Q_baked(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, m_vec, num_ks, num_ns, calib_params, pulse_params, outer_folder='', path='', prefix='data', meta_dict=None, plot=True, fit=True,):
        super().__init__(path=path, outer_folder=outer_folder,prefix=prefix, meta_dict=meta_dict)

        self.element = element
        self.num_ms = len(m_vec)
        self.num_ks = num_ks
        self.num_ns = num_ns
        self.calib_params = calib_params
        self.pulse_params = pulse_params
        self.plot = plot
        self.fit = fit

        self.m_vec_qua = [int(m_vec[i]) for i in range(int(self.num_ms))]
        self.m_vec = self.m_vec_qua
        self.num_ms = self.num_ms
        self.seed = rd.randint(1,1000000)

        self.config = meta_dict_to_config_6Q(self.meta_dict)

        # self.pickle_data()



    def generate_sequence(self, m):

        seq = []

        cayley = c1_table.flatten().tolist()
        inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
        current_state = 0

        for i in range(m):
            next_gate = rd.randint(0, 23)
            seq.append(next_gate)
            current_state = cayley[current_state*24+next_gate]

        inv_gate = inv_gates[current_state]

        seq.append(inv_gate)

        return seq

    def bake_sequence(self, seq):

        element = self.element

        self.baked_config = copy.deepcopy(self.config)

        pulse_params = self.pulse_params

        pulse_amp = pulse_params['pulse_amp']
        pulse_len = pulse_params['pulse_len']
        DRAG_beta = pulse_params['DRAG_beta']

        I = Omega_G(pulse_amp, pulse_len, pulse_len / 4)
        Q = Omega_G_dot(DRAG_beta * pulse_amp, pulse_len, pulse_len / 4)

        with baking(self.baked_config) as b:

            b.add_op('baked_op', element, [I, Q])

            for s in seq:

                if s == 0:
                    pass
                if s == 1:
                    b.play("X180", element)
                if s == 2:
                    b.play("Y180", element)
                if s == 3:
                    b.play("Y180", element)
                    b.play("X180", element)
                if s == 4:
                    b.play("X90", element)
                    b.play("Y90", element)
                if s == 5:
                    b.play("X90", element)
                    b.play("Ym90", element)
                if s == 6:
                    b.play("Xm90", element)
                    b.play("Y90", element)
                if s == 7:
                    b.play("Xm90", element)
                    b.play("Ym90", element)
                if s == 8:
                    b.play("Y90", element)
                    b.play("X90", element)
                if s == 9:
                    b.play("Y90", element)
                    b.play("Xm90", element)
                if s == 10:
                    b.play("Ym90", element)
                    b.play("X90", element)
                if s == 11:
                    b.play("Ym90", element)
                    b.play("Xm90", element)
                if s == 12:
                    b.play("X90", element)
                if s == 13:
                    b.play("Xm90", element)
                if s == 14:
                    b.play("Y90", element)
                if s == 15:
                    b.play("Ym90", element)
                if s == 16:
                    b.play("Xm90", element)
                    b.play("Y90", element)
                    b.play("X90", element)
                if s == 17:
                    b.play("Xm90", element)
                    b.play("Ym90", element)
                    b.play("X90", element)
                if s == 18:
                    b.play("X180", element)
                    b.play("Y90", element)
                if s == 19:
                    b.play("X180", element)
                    b.play("Ym90", element)
                if s == 20:
                    b.play("Y180", element)
                    b.play("X90", element)
                if s == 21:
                    b.play("Y180", element)
                    b.play("Xm90", element)
                if s == 22:
                    b.play("X90", element)
                    b.play("Y90", element)
                    b.play("X90", element)
                if s == 23:
                    b.play("Xm90", element)
                    b.play("Y90", element)
                    b.play("Xm90", element)

                # if s == 0:
                #     pass
                # if s == 1:
                #     b.play("X180", element)
                # if s == 2:
                #     b.play("X180", element)
                #     b.frame_rotation_2pi(-0.5, element)
                # if s == 3:
                #     b.frame_rotation_2pi(-0.5,element)
                # if s == 4:
                #     b.frame_rotation_2pi(0.25,element)
                #     b.play("X90", element)
                # if s == 5:
                #     # [cirq.S(cirq.LineQubit(0)), (cirq.X**0.5).on(cirq.LineQubit(0))]
                #     b.frame_rotation_2pi(-0.25, element)
                #     b.play("X90", element)
                # if s == 6:
                #     # [cirq.S(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))]
                #     b.frame_rotation_2pi(-0.25, element)
                #     b.play("Xm90", element)
                # if s == 7:
                #     # [(cirq.S**-1).on(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(0.25, element)
                #     b.play("Xm90", element)
                # if s == 8:
                #     # [(cirq.X**0.5).on(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                #     b.play("X90", element)
                #     b.frame_rotation_2pi(-0.25, element)
                # if s == 9:
                #     # [(cirq.X**-0.5).on(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))],
                #     b.play("Xm90", element)
                #     b.frame_rotation_2pi(0.25, element)
                # if s == 10:
                #     # [(cirq.X**0.5).on(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))],
                #     b.play("X90", element)
                #     b.frame_rotation_2pi(0.25, element)
                # if s == 11:
                #     # [(cirq.X**-0.5).on(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                #     b.play("Xm90", element)
                #     b.frame_rotation_2pi(-0.25, element)
                # if s == 12:
                #     b.play("X90", element)
                # if s == 13:
                #     b.play("Xm90", element)
                # if s == 14:
                #     b.play("Y90", element)
                # if s == 15:
                #     b.play("Ym90", element)
                # if s == 16:
                #     # [cirq.S(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(-0.25, element)
                # if s == 17:
                #     # [(cirq.S**-1).on(cirq.LineQubit(0))]
                #     b.frame_rotation_2pi(0.25, element)
                # if s == 18:
                #     # [(cirq.S ** -1).on(cirq.LineQubit(0)),
                #     #  (cirq.X ** 0.5).on(cirq.LineQubit(0)),
                #     #  (cirq.S ** -1).on(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(0.25, element)
                #     b.play("X90", element)
                #     b.frame_rotation_2pi(0.25, element)
                # if s == 19:
                #     # [cirq.S(cirq.LineQubit(0)),
                #     #  (cirq.X ** 0.5).on(cirq.LineQubit(0)),
                #     #  cirq.S(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(-0.25, element)
                #     b.play("X90", element)
                #     b.frame_rotation_2pi(-0.25, element)
                # if s == 20:
                #     # [cirq.Z(cirq.LineQubit(0)), (cirq.X**-0.5).on(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(-0.5, element)
                #     b.play("Xm90", element)
                # if s == 21:
                #     # [cirq.Z(cirq.LineQubit(0)), (cirq.X**0.5).on(cirq.LineQubit(0))],
                #     b.frame_rotation_2pi(-0.5, element)
                #     b.play("X90", element)
                # if s == 22:
                #     # [cirq.X(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))],
                #     b.play("X180", element)
                #     b.frame_rotation_2pi(-0.25, element)
                # if s == 23:
                #     # [cirq.X(cirq.LineQubit(0)), (cirq.S**-1).on(cirq.LineQubit(0))]]
                #     b.play("X180", element)
                #     b.frame_rotation_2pi(0.25, element)
                #
                # if s == 0:
                #     # I
                #     pass
                # if s == 1:
                #     # X180
                #     phased_XZ(element, b, 0, 1, 0, )
                # if s == 2:
                #     # Y180
                #     phased_XZ(element, b, 0.5, 1, 0, )
                # if s == 3:
                #     # Y180 X180
                #     phased_XZ(element, b, 0, 0, 1)
                # if s == 4:
                #     # X90 Y90
                #     phased_XZ(element, b, 0.5, 0.5, -0.5)
                # if s == 5:
                #     # X90 -Y90
                #     phased_XZ(element, b, 0.5, -0.5, 0.5)
                # if s == 6:
                #     # -X90 Y90
                #     phased_XZ(element, b, 0.5, 0.5, 0.5)
                # if s == 7:
                #     # -X90 -Y90
                #     phased_XZ(element, b, 0.5, -0.5, -0.5)
                # if s == 8:
                #     # Y90 X90
                #     phased_XZ(element, b, 0, 0.5, 0.5)
                # if s == 9:
                #     # Y90 -X90
                #     phased_XZ(element, b, 0, -0.5, -0.5)
                # if s == 10:
                #     # -Y90 X90
                #     phased_XZ(element, b, 0, 0.5, -0.5)
                # if s == 11:
                #     # -Y90 -X90
                #     phased_XZ(element, b, 0, -0.5, 0.5)
                # if s == 12:
                #     # X90
                #     phased_XZ(element, b, 0, 0.5, 0)
                # if s == 13:
                #     # -X90
                #     phased_XZ(element, b, 0, -0.5, 0)
                # if s == 14:
                #     # Y90
                #     phased_XZ(element, b, 0.5, 0.5, 0)
                # if s == 15:
                #     # -Y90
                #     phased_XZ(element, b, 0.5, -0.5, 0)
                # if s == 16:
                #     # -X90 Y90 X90
                #     phased_XZ(element, b, 0, 0, 0.5)
                # if s == 17:
                #     # -X90 -Y90 X90
                #     phased_XZ(element, b, 0, 0, -0.5)
                # if s == 18:
                #     # X180 Y90
                #     phased_XZ(element, b, 0.5, 0.5, 1)
                # if s == 19:
                #     # X180 -Y90
                #     phased_XZ(element, b, 0.5, -0.5, 1)
                # if s == 20:
                #     # Y180 X90
                #     phased_XZ(element, b, 0, 0.5, 1)
                # if s == 21:
                #     # Y180 -X90
                #     phased_XZ(element, b, 0, -0.5, 1)
                # if s == 22:
                #     # X90 Y90 X90
                #     phased_XZ(element, b, 0.25, 1, 0)
                # if s == 23:
                #     # -X90 Y90 -X90
                #     phased_XZ(element, b, -0.25, 1, 0)

        return b


    def make_prog(self, m):

        reset_ticks = self.meta_dict['global_params']['reset_ticks']

        seq = self.generate_sequence(m)

        b = self.bake_sequence(seq)

        with program() as prog:

            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < self.num_ns, n + 1):

                wait(reset_ticks, self.element)

                reset_frame(self.element)

                align()

                # play('X180', self.element)

                b.run()

                align()

                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                save(I, I_stream)
                save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog,seq


    def make_amp_tune_prog(self, a):

        reset_ticks = self.meta_dict['global_params']['reset_ticks']

        element = self.element

        self.baked_config = copy.deepcopy(self.config)

        pulse_params = self.pulse_params

        pulse_amp = a
        pulse_len = pulse_params['pulse_len']
        DRAG_beta = pulse_params['DRAG_beta']

        I = Omega_G(pulse_amp, pulse_len, pulse_len / 4)
        Q = Omega_G_dot(DRAG_beta * pulse_amp, pulse_len, pulse_len / 4)

        with baking(self.baked_config) as b:

            b.add_op('baked_op', element, [I, Q])

            phased_XZ(0, 0.5, 0, element, b)

        with program() as prog:

            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < self.num_ns, n + 1):

                wait(reset_ticks, self.element)

                reset_frame(self.element)

                align()

                b.run()

                align()

                I, Q = measure_1q(self.meta_dict[self.element], I, Q)

                save(I, I_stream)
                save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog


    def run_mk_programs(self):

        times = []
        start_time = time.time()
        data = np.zeros([self.num_ks, self.num_ms, self.num_ns])
        sequences = []

        for k in range(self.num_ks):
            print(k)
            times.append(time.time()-start_time)
            print(times)
            seq_list = []
            for m, M in enumerate(self.m_vec):

                prog,seq = self.make_prog(M)
                seq_list.append(seq)

                job = self.acquire(prog, config=self.baked_config)

                # ss_array = self.job_to_shots_1q(job, [self.num_ns], self.calib_params)
                ss_array = self.job_to_shots_1q(job, [self.num_ns], self.calib_params)

                data[k][m] = ss_array

                # for n in range(self.num_ns):
                #     data[k][m][n] = find_state([I[n], Q[n]], IQ_0, IQ_1, IQ_2)

            sequences.append(seq_list)


        self.seqs = sequences

        self.data = [self.m_vec, data]

    def amp_cal(self, a_min, a_max, num_as):

        data = np.zeros([num_as, self.num_ns])
        a_vec = np.linspace(a_min, a_max, num_as)

        for i,a in enumerate(a_vec):

            prog = self.make_amp_tune_prog(a)

            job = self.acquire(prog, config=self.baked_config)

            ss_array = self.job_to_shots_1q_3state(job, [self.num_ns], self.calib_params)

            data[i] = ss_array

        self.amp_cal_data = [a_vec, data]


    def analyze(self):

        ss_array = self.data[1]

        probs = np.zeros([3, self.num_ms])

        num_shots_per_m = self.num_ks * self.num_ns

        probs_k = np.zeros([3, self.num_ks, self.num_ms])

        for k in range(self.num_ks):
            for n in range(self.num_ns):
                for m in range(self.num_ms):
                    probs[int(ss_array[k][m][n])][m] += 1 / num_shots_per_m
                    probs_k[int(ss_array[k][m][n])][k][m] += 1 / self.num_ns

        probs_k_std = [np.std(probs_k[i], axis=0) for i in range(3)]


        self.data.append(probs)
        self.data.append(probs_k_std)

        def fitRB(x,a,p,b):
            return a*p**x+b

        self.gate_fidel = 0

        # fit_array = ss_array.mean(axis=(0,2))
        # fit_errs = np.array([np.std(ss_array[:,i,:].mean(axis=1)) for i in range(self.num_ms)])

        # return_prob = 1 - fit_array

        if self.fit:

            # try:
            popt, pcov = sp.optimize.curve_fit(fitRB, self.m_vec, probs[0],
                                               [0.3, 0.99, 0.5])

            popt_leak, pcov_leak = sp.optimize.curve_fit(fitRB, self.m_vec, probs[2],
                                                         [-0.1, 0.99, 0.1], maxfev=5000)
            # except:
            #
            #     popt = [0.5,1,0.5]
            #     pcov = [[1,1,1],[1,1,1],[1,1,1]]
            #
            #     popt_leak = [0.5, 1, 0.5]
            #     pcov_leak = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


            a = popt[0]
            p = popt[1]
            p_err = np.sqrt(pcov[1][1])
            b = popt[2]

            self.clif_fidel = 1 - 0.5 * (1 - p)
            self.gate_fidel = 1 - 0.5 * (1 - p) / (20/24) #1.875
            self.gate_fidel_err = p_err * 0.5 / (20/24) #1.875
            self.amp = a
            self.off = b

            self.leak_rate = (1-popt_leak[1])*popt_leak[2] / (20/24)
            self.leak_rate_err = np.sqrt(pcov_leak[1][1])*popt_leak[2] / (20/24)

            print(self.gate_fidel)
            print(self.leak_rate)

        # for k in range(self.num_ks):
        #     for m in range(self.num_ms):
        #         if ss_array.mean(axis=2)[k][m] > 0.5:
        #             print(self.seqs[k][m])

        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            # ax.errorbar(self.m_vec, probs[0],yerr=fit_errs,fmt= 'o')
            ax.errorbar(self.m_vec, probs[0], yerr=probs_k_std[0], fmt='o')
            ax.errorbar(self.m_vec, probs[1], yerr=probs_k_std[1], fmt='o')
            # ax.errorbar(self.m_vec, probs[2] * 1, yerr=1*probs_k_std[2], fmt='o')
            if self.fit:
                ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]), fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt))
                # ax.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                #         1*fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_leak))
            ax.set_xlabel('Num Cliffords')
            ax.set_ylabel('Return probability')
            q_name = self.element
            ax.set_title(f'RB on {q_name}, F = {self.gate_fidel:.4f} +/- {self.gate_fidel_err:.4f}')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)



    def analyze_amp_cal(self):

        ss_array = self.amp_cal_data[1]
        a_vec = self.amp_cal_data[0]
        num_as = len(a_vec)

        probs = np.zeros([3, num_as])

        for a in range(num_as):
            for n in range(self.num_ns):
                probs[int(ss_array[a][n])][a] += 1 / self.num_ns


        if self.plot:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            # ax.errorbar(self.m_vec, probs[0],yerr=fit_errs,fmt= 'o')
            ax.plot(a_vec, probs[0], 'o')
            ax.plot(a_vec, probs[1], 'o')
            ax.plot(a_vec, probs[2], 'o')
            ax.set_xlabel('Amplitude (V)')
            ax.set_ylabel('Probability')
            ax.set_title(f'Amplitude calibration')
            plt.savefig(self.iname, bbox_inches="tight")

            plt.close(fig)


if __name__ == '__main__':
    outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
    pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

    # Get metaDict
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)

    m_vec = [2,4,8,16,32,64,128,256,512,1024]
    num_ks = 200
    num_ns = 10

    # calib_params = [[0,0],[1,1],[2,2]]

    q = 'q4'
    SS = SingleShot1Q(q, 1000, meta_dict=meta_dict, outer_folder=outer_folder, path="RB",
                      suffix="read_cal")

    calib_params = SS.calib_params



    pulse_params_2 = {'pulse_amp': meta_dict['q4']['pi_amp'],
                      'pulse_len': meta_dict['global_params']['gauss_len'],
                      'DRAG_beta': 0}

    RB_1 = RB1Q_baked(q,m_vec,num_ks,num_ns,calib_params,pulse_params_2, path="RB", outer_folder=outer_folder,meta_dict=meta_dict)

    RB_1.run_mk_programs()
    RB_1.analyze()

    # RB_1.simulate(RB_1.make_prog(2048),['3','7'],1000,[[200,1000]],config=RB_1.baked_config)
