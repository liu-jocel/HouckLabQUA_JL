from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_3q
from LabCode.Config.config_helpers import Omega_G, Omega_G_dot
import scipy as sp
import scipy.optimize
import random as rd
import time
from qualang_tools.bakery.bakery import baking
from LabCode.Config.config_helpers import meta_dict_to_config_2Q
from LabCode.Helpers.gate_macros import phased_XZ, UGate, clifford_1Q, PMW2
from LabCode.Helpers.RB_helpers import generate_rb_phasedXZ, generate_rb_UGate, generate_rb_Unitary, PMW2_matrix
from LabCode.Experiments.m_single_shot_discrim import SingleShotDiscrim
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Helpers.ss_helpers import Discrim_G_C2Q3L
import os
import datetime


class RB1Q_baked(ExperimentClass):
    """
    RB experiement
    """

    def __init__(self, element, m_vec, num_ks, num_ns, gates, discrim_class, outer_folder='', path='', prefix='data', suffix="data", meta_dict=None, plot=True, fit=True, confusion = np.identity(4)):
        super().__init__(path=path, outer_folder=outer_folder,prefix=prefix, suffix=suffix, meta_dict=meta_dict)

        self.element = element
        self.num_ms = len(m_vec)
        self.num_ks = num_ks
        self.num_ns = num_ns
        # self.gate_dict = gate_dict
        self.gates = gates
        self.plot = plot
        self.fit = fit
        self.discrim = discrim_class
        self.confusion = confusion

        self.m_vec_qua = [int(m_vec[i]) for i in range(int(self.num_ms))]
        self.m_vec = self.m_vec_qua
        self.num_ms = self.num_ms
        self.seed = rd.randint(1,1000000)
        self.dimension = 2

        self.config = meta_dict_to_config_2Q(self.meta_dict)


    def generate_sequence(self, m, gate_dict):
        """
        gate_dict:{
            "type" : PhXZ or UGate
            "angles" : x,z,a or theta, phi, lam respectively
        }

        "rb_dict" is a dictionary with three keys:
            "gate": Contains the function for the arbitrary 1Q unitary
            "sequence" : RB sequence
            "reference_inverse" : inverse unitary for reference sequence in parametrization style of gate_type
            "interleaved_inverse" : inverse unitary for interleaved sequence in parametrization style of gate_type
        """
        gate = gate_dict["gate_func"]
        gate_unitary = gate_dict["gate_unitary"]
        gate_angles = gate_dict["angles"]

        if type(gate)==type(phased_XZ):
            rb_dict = generate_rb_phasedXZ(m, *gate_angles)

        elif type(gate)==type(UGate):
            rb_dict = generate_rb_UGate(m, *gate_angles)
        else:
            rb_dict = generate_rb_Unitary(m, gate_dict)
        return rb_dict

    def bake_sequence(self, rb_dict, interleaved=False):

        element = self.element
        seq = rb_dict["sequence"]


        self.baked_config = copy.deepcopy(self.config)

        pulse_amp = self.meta_dict[element]['pi_amp']
        pulse_len = self.meta_dict["global_params"]['gauss_len']
        DRAG_beta = self.meta_dict[element]['DRAG_beta']

        I = Omega_G(pulse_amp, pulse_len, pulse_len / 4)
        Q = Omega_G_dot(DRAG_beta * pulse_amp, pulse_len, pulse_len / 4)

        with baking(self.baked_config) as b:
            b.add_op('baked_op', element, [I, Q])
            if interleaved:
                for s in seq:
                    clifford_1Q(s, element, b)
                    rb_dict["gate"](element, b, *rb_dict["gate_args"])
                rb_dict["gate"](element, b, *rb_dict["interleaved_inverse"])
            else:
                for s in seq:
                    clifford_1Q(s, element, b)
                rb_dict["gate"](element, b, *rb_dict["reference_inverse"])
        return b

    def make_prog(self, rb_dict, interleaved):

        reset_ticks = self.meta_dict['global_params']['reset_ticks']

        b = self.bake_sequence(rb_dict, interleaved=False)

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

            with for_(n, 0, n < self.num_ns, n + 1):

                wait(reset_ticks, self.element)

                reset_frame(self.element)

                align()

                b.run()

                align()

                I1, Q1, I2, Q2, Ic, Qc = measure_3q(self.meta_dict["q1"], self.meta_dict["q2"],
                                                    self.meta_dict["qc"],
                                                    I1, Q1, I2, Q2, Ic, Qc)

                save(I1, I1_stream)
                save(Q1, Q1_stream)
                save(I2, I2_stream)
                save(Q2, Q2_stream)
                save(Ic, Ic_stream)  #
                save(Qc, Qc_stream)  #

            with stream_processing():
                I1_stream.save_all("I1")
                Q1_stream.save_all("Q1")
                I2_stream.save_all("I2")
                Q2_stream.save_all("Q2")
                Ic_stream.save_all("Ic")
                Qc_stream.save_all("Qc")
        return prog


    def run_mk_programs(self):
        # Todo: add data caching

        start_time = time.time()
        out_data = {gd["name"]: {
            "times": [],
            "data_ref": [],
            "data_int": [],
            'sequence': [],
        } for gd in self.gates}

        for k in range(self.num_ks):
            print(k)
            for gate_dict in self.gates:
                gate_name = gate_dict["name"]
                current_output = out_data[gate_name]
                current_output["times"].append(time.time()-start_time)
                seq_list = []
                temp_data_ref = []
                temp_data_int = []
                for m in self.m_vec:
                    rb_dict = self.generate_sequence(m, gate_dict)
                    seq_list.append((rb_dict["sequence"]))

                    # Run Reference
                    prog_ref = self.make_prog(rb_dict, interleaved=False)
                    job_ref = self.acquire(prog_ref, config=self.baked_config)
                    counts_dict_ref = super().job_to_shots_arb(job_ref, [self.num_ns, 1], self.discrim)
                    temp_data_ref.append([value[0] for value in counts_dict_ref.values()])  # return probability

                    # Run Interleaved
                    prog_int = self.make_prog(rb_dict, interleaved=True)
                    job_int = self.acquire(prog_int, config=self.baked_config)
                    counts_dict_int = super().job_to_shots_arb(job_int, [self.num_ns, 1], self.discrim)
                    temp_data_int.append([value[0] for value in counts_dict_int.values()]) # return probability

                current_output["sequence"].append(seq_list)
                current_output["data_ref"].append(temp_data_ref)
                current_output["data_int"].append(temp_data_int)

                with open(self.pname, 'wb') as f:
                    pickle.dump(out_data, f)

        self.data = out_data

    def analyze(self):
        # ToDO: fix analysis to work with dictionary
        data_dict = self.data
        m_vec = self.m_vec
        dimension = self.dimension

        out_dict = {}
        for gate_name in data_dict:

            out_dict[gate_name] = {}

            gate_dict = data_dict[gate_name]
            seq = gate_dict["sequence"]
            data_ref = np.array(gate_dict["data_ref"])
            data_int = np.array(gate_dict["data_int"])

            avg_data_ref = data_ref.mean(axis=0)
            std_data_ref = data_ref.std(axis=0)[:,0]
            return_prob_ref = avg_data_ref[:, 0]
            avg_data_int = data_int.mean(axis=0)
            std_data_int = data_int.std(axis=0)[:,0]
            return_prob_int = avg_data_int[:,0]

            popt_ref = 0
            pcov_ref = 0
            popt_int = 0
            pcov_int = 0

            if self.fit:
                def fitRB(x, a, p, b):
                    return a * p ** x + b

                try:
                    popt_ref, pcov_ref = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_ref,
                                                       [0.3, 0.99, 0.5], maxfev=5000)

                    popt_int, pcov_int = sp.optimize.curve_fit(fitRB, self.m_vec, return_prob_int,
                                                                 [0.3, 0.99, 0.5], maxfev=5000)
                except:
                    popt_ref = [0.5, 1, 0.5]
                    pcov_ref = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

                    popt_int = [0.5, 1, 0.5]
                    pcov_int = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

                p_ref = popt_ref[1]
                p_ref_error = np.sqrt(pcov_ref[1][1])
                p_int = popt_int[1]
                p_int_error = np.sqrt(pcov_int[1][1])

                error_per_gate = (dimension - 1)/dimension * (1 - p_int/p_ref)
                error_per_gate_error = (dimension - 1)/dimension * (p_ref_error/p_ref + p_int_error/p_int) * error_per_gate

            if self.plot:
                datetime_now = datetime.datetime.now()
                date_string = datetime_now.strftime('%Y_%m_%d')
                time_string = datetime_now.strftime('%H_%M_%S')
                image_name = os.path.join(self.outerFolder + self.path, self.path + "_" + date_string, self.path + "_"+
                                          time_string + "_" + gate_name + '.png')
                fig, ax = plt.subplots(1,1)
                plt.plot(m_vec, return_prob_ref, 'bo', label='Ref')
                plt.plot(m_vec, return_prob_int, 'ro', label='Interleaved')
                ax.errorbar(m_vec, return_prob_ref, yerr=std_data_ref/np.sqrt(num_ks), fmt='bo')
                ax.errorbar(m_vec, return_prob_int, yerr=std_data_int/np.sqrt(num_ks), fmt='ro')
                plt.legend()

                if self.fit:
                    plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                             fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_ref), 'b', label=None)
                    plt.plot(np.linspace(self.m_vec[0], self.m_vec[-1]),
                             fitRB(np.linspace(self.m_vec[0], self.m_vec[-1]), *popt_int), 'r', label=None)
                    plt.title(f'IRB on {gate_name}, Error Per gate: = {error_per_gate:.7f}+/-{error_per_gate_error:.7f}')

                plt.xlabel('Num Cliffords')
                plt.ylabel('Return probability')

                plt.savefig(image_name, bbox_inches="tight")
                plt.close(fig)

            out_dict[gate_name]["gate"] = gate_dict
            out_dict[gate_name]["sequences"] = seq
            out_dict[gate_name]["data_ref"] = (data_ref, std_data_ref)
            out_dict[gate_name]["data_int"] = (data_int, std_data_int)
            out_dict[gate_name]["fit_ref"] = (popt_ref, pcov_ref)
            out_dict[gate_name]["fit_int"] = (popt_int, pcov_int)
            gate_dict["error_per_gate"] = (error_per_gate, error_per_gate_error)
        return out_dict
        # return {
        #     "gate": self.gate_dict,
        #     "sequences": sequences,
        #     "m_vec": m_vec,
        #     "k": self.num_ks,
        #     "data_ref": (data_ref, std_data_ref),
        #     "data_int": (data_int, std_data_int),
        #     "fit_ref": (popt_ref, pcov_ref),
        #     "fit_int": (popt_int, pcov_int),
        #     "error_per_gate": (error_per_gate, error_per_gate_error)
        # }


if __name__ == '__main__':
    #
    outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
    pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

    # Get metaDict
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)

    m_vec = [2 , 44, 88, 132, 176, 220, 256,]
    num_ks = 100
    num_ns = 10
    meta_dict['qc']['read_amp'] = 0
    meta_dict['q1']['read_amp'] = 0

    qubit = 'q4'
    gates = [
     {
        "name": f"PhXZ_{x}_q4",
        "gate_unitary": None,
        "gate_func": phased_XZ,
        "angles": [float(np.random.rand()), 1/x, float(np.random.rand())]
     } for x in [1,]]

    state_names = {'000', '010'}
    POVM = {
        f'{qubit}_in_0': ['000'],
        f'{qubit}_in_1': ['100']
    }

    SS = SingleShotDiscrim('q4', 'q3', 'q1', 1000, state_names, POVM, Discrim_G_C2Q3L,
                           meta_dict=meta_dict, outer_folder=outer_folder, path='RB',
                           suffix='discrim')  # 'qc' added

    discrim = SS.discrim
    print(f" Readout Fidelity: {discrim.readout_fidelity}")
    print("####################################################")
    RB_1 = RB1Q_baked(qubit, m_vec, num_ks, num_ns, gates, discrim_class=discrim, path="RB", fit=True,
                      outer_folder=outer_folder, meta_dict=meta_dict,)
    RB_1.run_mk_programs()
    data = RB_1.analyze()
    with open(RB_1.pname, 'wb') as f:
        pickle.dump(data, f)

##################################################################################
##################################################################################

        #
        outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
        pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

        # Get metaDict
        with open(pathPickle, 'rb') as f:
            meta_dict = pickle.load(f)

        m_vec = [2, 44, 88, 132, 176, 220, 256, ]
        num_ks = 100
        num_ns = 10
        meta_dict['qc']['read_amp'] = 0
        meta_dict['q1']['read_amp'] = 0

        qubit = 'q2'
        gates = [
            {
                "name": f"UGate_{x}_q2",
                "gate_unitary": None,
                "gate_func": UGate,
                "angles": [float(np.random.rand()), 1 / x, float(np.random.rand())]
            } for x in [1, 2, 4, 8]]

        state_names = {'000', '010'}
        POVM = {
            f'{qubit}_in_0': ['000'],
            f'{qubit}_in_1': ['010']
        }

        SS = SingleShotDiscrim('q1', 'q2', 'qc', 1000, state_names, POVM, Discrim_G_C2Q3L,
                               meta_dict=meta_dict, outer_folder=outer_folder, path='RB',
                               suffix='discrim')  # 'qc' added

        discrim = SS.discrim
        print(f" Readout Fidelity: {discrim.readout_fidelity}")
        print("####################################################")
        RB_1 = RB1Q_baked(qubit, m_vec, num_ks, num_ns, gates, discrim_class=discrim, path="RB", fit=True,
                          outer_folder=outer_folder, meta_dict=meta_dict, )
        RB_1.run_mk_programs()
        data = RB_1.analyze()
        with open(RB_1.pname, 'wb') as f:
            pickle.dump(data, f)


##################################################################################
##################################################################################
    outer_folder = 'Z://t1Team/Data/2024-08-09_cooldown/TATQ01-si_01_qm/'
    pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

    # Get metaDict
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)

    m_vec = [2, 44, 88, 132, 176, 220, 256]
    num_ks = 100
    num_ns = 10
    # meta_dict['qc']['read_amp'] = 0
    # meta_dict['q1']['read_amp'] = 0

    qubit = 'q4'
    gates = [
     {
        "name": f"PMW2_{x}_q4",
        "gate_unitary": PMW2_matrix,
        "gate_func": PMW2,
        "angles": [float(np.random.rand()), 1/x, float(np.random.rand())]
     } for x in [1]]

    # state_names = {'000', '010'}
    # POVM = {
    #     f'{qubit}_in_0': ['000'],
    #     f'{qubit}_in_1': ['010']
    # }

    SS = SingleShot1Q(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path=path,
                      suffix="read_cal")

    calib_params = SS.calib_params

    # SS = SingleShotDiscrim('q1', 'q2', 'qc', 1000, state_names, POVM, Discrim_G_C2Q3L,
    #                        meta_dict=meta_dict, outer_folder=outer_folder, path='RB',
    #                        suffix='discrim')  # 'qc' added
    #
    # discrim = SS.discrim
    print(f" Readout Fidelity: {discrim.readout_fidelity}")
    print("####################################################")
    RB_1 = RB1Q_baked(qubit, m_vec, num_ks, num_ns, gates, discrim_class=discrim, path="RB", fit=True,
                      outer_folder=outer_folder, meta_dict=meta_dict,)
    RB_1.run_mk_programs()
    data = RB_1.analyze()
    with open(RB_1.pname, 'wb') as f:
        pickle.dump(data, f)

    #Q2
    # with open(pathPickle, 'rb') as f:
    #     meta_dict = pickle.load(f)
    #
    # m_vec = [2, 22, 44, 88, 110, 132]# 44, 88, 132, 176, 220, 256,
    # num_ks = 100
    # num_ns = 10
    # meta_dict['qc']['read_amp'] = 0
    # meta_dict['q1']['read_amp'] = 0
    #
    # qubit = 'q2'
    # gates = [
    #  {
    #     "name": f"PhXZ{x}_q2",
    #     "gate_unitary": PMW2_matrix,
    #      "gate_func" : PMW2,
    #     "angles": [float(np.random.rand()), 1/x, float(np.random.rand())]
    #  } for x in [1]] #2, 4, 8,16
    #
    # state_names = {'000', '010'}
    # POVM = {
    #     f'{qubit}_in_0': ['000'],
    #     f'{qubit}_in_1': ['010']
    # }
    #
    # SS = SingleShotDiscrim('q1', 'q2', 'qc', 1000, state_names, POVM, Discrim_G_C2Q3L,
    #                        meta_dict=meta_dict, outer_folder=outer_folder, path='RB',
    #                        suffix='discrim')  # 'qc' added
    #
    # discrim = SS.discrim
    # print(f" Readout Fidelity: {discrim.readout_fidelity}")
    # print("####################################################")
    # RB_1 = RB1Q_baked(qubit, m_vec, num_ks, num_ns, gates, discrim_class=discrim, path="RB", fit=True,
    #                   outer_folder=outer_folder, meta_dict=meta_dict,)
    # RB_1.run_mk_programs()
    # data = RB_1.analyze()
    # with open(RB_1.pname, 'wb') as f:
    #     pickle.dump(data, f)
    #
