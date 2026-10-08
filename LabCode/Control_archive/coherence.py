import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode.Config.config_helpers import meta_dict_to_config_6Q
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments_Archive.m_T1 import T1
from LabCode.Experiments_Archive.m_T2R import T2R
from LabCode.Experiments_Archive.m_ZZ import ZZ
from LabCode.Experiments_Archive.m_T2E import T2E
from LabCode.Experiments_Archive.m_T2CPMG import T2CPMG
from LabCode.Experiments.m_transmission import Transmission
from LabCode.Experiments.m_qubit_spec import QubitSpec
from LabCode.Experiments_Archive.m_Rabi import Rabi
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from qm.qua import *
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

meta_dict = make_meta_dict()
outer_folder = meta_dict["outer_folder"]
# YOKOGS200.main(0.52)

qubit_list = ['q4']
### for pickling data - cheap hack
pickled_data = []
start = time.time()
pickled_data.append(time.ctime(start))
file_name = outer_folder + 'coherence/coherence.pkl'
###

while True: # remember removing break while

    data_dict = {}
    data_dict['timestamp'] = time.time() - start
    for q in qubit_list:
        path = q
        ### for pickling data - cheap hack

        ###

        Q = meta_dict[q]

        config = meta_dict_to_config_6Q(meta_dict)

        LOs = [Q['q_LO'], Q['r_LO']]

        # set_LO_parameters(LOs)
        # always set the LOs to the same value for readout while multiplexing

        qubits = [Q['q_name']]

        do_readout_cal = 1
        do_transmission = 0
        do_qubit_spec = 0
        do_rabi_IQ = 0
        do_rabi_SS = 1
        do_rabi_time = 0
        do_fine_tune = 0
        do_T1 = 0
        do_T2R = 0
        do_T2E = 0
        do_T2CPMG = 0
        do_ZZ = 0
        do_T1_sweep = 0

        # Single shot parameters
        shots_SS = [1000]

        # Transmission parameters
        # f_min_trans = [30e6, 130e6, 230e6]
        # f_max_trans = [70e6, 170e6, 270e6]
        f_min_trans = [-110 * u.MHz,]
        f_max_trans = [-100 * u.MHz, ]
        f_step_trans = [0.1 * u.MHz, ]
        shots_trans = [200]

        # Qubit spectroscopy parameters
        # resonator = meta_dict['q1']['r_name']
        # freq = 30.1e6
        # update_frequency(resonator, freq)
        f_min_spec = [-190 * u.MHz]
        f_max_spec = [-180 * u.MHz]
        f_step_spec = [.1 * u.MHz]
        shots_spec = [200]

        # Rabi parameters
        # t_min_rabi = [16e-9,]
        # t_max_rabi = [4000e-9,]
        # num_t_rabi = [451]
        pulse_type = "X90"
        num_pi_pulses = [10000]
        a_min_rabi = [.95]
        a_max_rabi = [1.05]
        num_a_rabi = [11]
        freq_range_rabi = [0 * u.MHz] # 150
        num_freqs_rabi = [1] # 41
        shots_rabi = [1000]

        # T1 parameters
        t_min_t1 = [1e-6]
        t_max_t1 = [4000e-6]  #800 800 80
        num_t_t1 = [31]
        shots_t1 = [2000] #1000

        # T2R parameters
        t_min_t2r = [100e-9]
        t_max_t2r = [200e-6]
        num_t_t2r = [201]
        shots_t2r = [200]
        detuning_t2r = [.2e6]

        # T2E
        t_min_t2e = [16e-6]
        t_max_t2e = [100e-6]
        num_t_t2e = [31]
        shots_t2e = [2000]

        # T2CPMG
        t_min_t2CPMG = [100e-9]
        t_max_t2CPMG = [100e-6]
        num_t_t2CPMG = [31]
        numpi_CPMG = [128,]            #t_min/numpi must be greater than 16 ns
        shots_t2CPMG = [2000]

        # ZZ
        t_min_ZZ = 100e-9
        t_max_ZZ = 0.5e-6
        num_t_ZZ = 50
        shots_ZZ = 500
        detuning_ZZ = 4e6


        if do_readout_cal:
            time.sleep(10)
            calib_params = []
            for i,q in enumerate(qubits):
                SS = SingleShot1Q(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path=path,
                                    suffix="read_cal", repeats=1)
                calib_params.append(SS.calib_params)

                data_dict[q + '_calib_params'] = SS.calib_params

        if do_transmission:
            for i,q in enumerate(qubits):
                exp = Transmission(q, f_min_trans[i], f_max_trans[i], f_step_trans[i], shots_trans[i], meta_dict=meta_dict, outer_folder=outer_folder,
                                   path=path, suffix="trans",)

        if do_qubit_spec:
            for i,q in enumerate(qubits):
                set_LO_parameters([meta_dict[q]['q_LO']])
                exp = QubitSpec(q, f_min_spec[i], f_max_spec[i], f_step_spec[i], shots_spec[i], meta_dict=meta_dict, outer_folder=outer_folder,
                                path=path, suffix="spec")

        if do_rabi_IQ or do_rabi_SS:
            for i, q in enumerate(qubits):
                set_LO_parameters([meta_dict[q]['q_LO']])
                chevron_data = np.zeros([num_freqs_rabi[i], num_a_rabi[i]])
                freq_range = np.linspace(-freq_range_rabi[i]/2,freq_range_rabi[i]/2, num_freqs_rabi[i])
                LO_freqs = meta_dict[q]['q_LO']['LO_freq'] + freq_range
                start_time = time.time()

                for j in range(num_freqs_rabi[i]):

                    print(j, time.time()-start_time)
                    meta_dict[q]['q_LO']['LO_freq'] = LO_freqs[j]
                    set_LO_parameters([meta_dict[q]['q_LO']])

                    if num_freqs_rabi[i] == 1: # Use native plotting in experiment if you just want a line cut
                        if do_rabi_IQ:
                            exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], None,pulse_type=pulse_type, num_pi_pulses=num_pi_pulses[i], meta_dict=meta_dict, outer_folder=outer_folder,
                                   path=path, suffix="rabi_IQ")
                        elif do_rabi_SS:
                            exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], calib_params[i], pulse_type=pulse_type, num_pi_pulses=num_pi_pulses[i], meta_dict=meta_dict, outer_folder=outer_folder,
                                   path=path, suffix="rabi_SS", ss=True)
                    else: # Do plotting here if you want a chevron
                        if do_rabi_IQ:
                            exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], None, pulse_type=pulse_type, num_pi_pulses=num_pi_pulses[i], meta_dict=meta_dict,
                                     outer_folder=outer_folder, path=path, suffix='rabi_IQ', plot=False)
                        elif do_rabi_SS:
                            exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], calib_params[i], pulse_type=pulse_type, num_pi_pulses=num_pi_pulses[i],  meta_dict=meta_dict,
                                     outer_folder=outer_folder, path=path, suffix='rabi_SS', plot=False, ss=True)

                        if j==0:
                            name = exp.iname[:-8]

                        chevron_data[j] = exp.data[1]
                        if num_freqs_rabi[i] > 1:
                            fig, ax = plt.subplots(1,1,constrained_layout=True)
                            pcm1 = ax.pcolormesh(exp.amp_vec, freq_range/1e6, chevron_data)
                            ax.set_xticks(exp.amp_vec)
                            ax.set_xticklabels(np.round(exp.amp_vec, decimals=4))
                            ax.tick_params(axis='x', rotation=45)
                            ax.set_yticks(freq_range/1e6)
                            ax.set_yticklabels(np.round(freq_range/1e6, decimals=2))
                            ax.set_xlabel("Amplitude Scale Factor")
                            ax.set_ylabel("Frequency Detuning (MHz)")
                            fig.colorbar(pcm1, ax=ax)
                            plt.savefig(name+'chevron', bbox_inches='tight')
                            plt.close()

        # if do_rabi_time:
        #     for i,q in enumerate(qubits):
        #         set_LO_parameters([meta_dict[q]['q_LO']])
        #         chevron_data = np.zeros([num_freqs_rabi[i],num_t_rabi[i]])
        #         LO_freqs = meta_dict[q]['q_LO']['LO_freq'] + np.linspace(-freq_range_rabi[i]/2,freq_range_rabi[i]/2,num_freqs_rabi[i])
        #         start_time = time.time()
        #
        #         for j in range(num_freqs_rabi[i]):
        #
        #             print(j, time.time()-start_time)
        #             meta_dict[q]['q_LO']['LO_freq'] = LO_freqs[j]
        #             set_LO_parameters([meta_dict[q]['q_LO']])
        #
        #             if num_freqs_rabi[i] == 1: # Use native plotting in experiment if you just want a line cut
        #                 exp = Rabi_time(q, t_min_rabi[i], t_max_rabi[i], num_t_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict, outer_folder=outer_folder,
        #                        path='1Q_Char', suffix=q)
        #             else: # Do plotting here if you want a chevron
        #                 exp = Rabi_time(q, t_min_rabi[i], t_max_rabi[i], num_t_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict,
        #                          outer_folder=outer_folder, path='1Q_Char', suffix='', plot=False)
        #                 if j==0:
        #                     name = exp.iname[:-8]
        #
        #                 chevron_data[j] = exp.data[1]
        #                 if num_freqs_rabi[i] > 1:
        #                     fig, ax = plt.subplots(1,1,constrained_layout=True)
        #                     pcm1 = ax.pcolormesh(exp.t_vec, LO_freqs, chevron_data)
        #                     fig.colorbar(pcm1, ax=ax)
        #                     plt.savefig(name+'chevron', bbox_inches='tight')
        #                     plt.close()
        #         with open(exp.pname, 'wb') as f:
        #                 pickle.dump(chevron_data, f)
        # #
        if do_T1:
            # time.sleep(30)
            for i,q in enumerate(qubits):
                set_LO_parameters([meta_dict[q]['q_LO']])
                exp = T1(q, t_min_t1[i], t_max_t1[i], num_t_t1[i], shots_t1[i], calib_params[i],
                         meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T1", fit=True, save=True) #calib_params[i]

            #for pickling data - cheap hack
            data_dict[q + '_T1'] = exp.data
            ###



        if do_T2R:
            for i,q in enumerate(qubits):
                set_LO_parameters([meta_dict[q]['q_LO']])
                exp = T2R(q, t_min_t2r[i], t_max_t2r[i], num_t_t2r[i], shots_t2r[i], detuning_t2r[i], calib_params[i],
                          meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T2R",freq_cal=False, save=True)
            data_dict[q + '_T2R'] = exp.data


        if do_T2E:
            for i,q in enumerate(qubits):
                set_LO_parameters([meta_dict[q]['q_LO']])
                exp = T2E(q, t_min_t2e[i], t_max_t2e[i], num_t_t2e[i], shots_t2e[i], calib_params[i],
                                  meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T2E", save=True)
            ### for pickling data - cheap hack
            data_dict[q + '_T2E'] = exp.data

        if do_T2CPMG:
            for i,q in enumerate(qubits):
                for n in numpi_CPMG:
                    set_LO_parameters([meta_dict[q]['q_LO']])
                    exp = T2CPMG(q, t_min_t2CPMG[i], t_max_t2CPMG[i], num_t_t2CPMG[i], n, shots_t2CPMG[i], calib_params[i],
                             meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix=f"T2CPMG_{n}", save=True, fit=False)

    pickled_data.append(data_dict)
    with open(file_name, 'wb') as f:
        pickle.dump(pickled_data, f)
    ###

        if do_ZZ:
            for i,q in enumerate(qubits):
                exp = ZZ(qubits[i],qubits[(i+1)%2],t_min_ZZ,t_max_ZZ,num_t_ZZ,shots_ZZ,detuning_ZZ,calib_params[i],
                         outer_folder=outer_folder,path=path, suffix="ZZ", meta_dict=meta_dict, plot=True, fit=True, save=True)

        t1_amps = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        if do_T1_sweep:
            for a in t1_amps:
                meta_dict['c1']['pulse_amp'] = a
                for i,q in enumerate(qubits):
                    set_LO_parameters([meta_dict[q]['q_LO']])
                    exp = T1(q, t_min_t1[i], t_max_t1[i], num_t_t1[i], shots_t1[i], calib_params[i],
                             meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T1"+f"_amp_{a}", save=True)
    break