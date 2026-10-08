import pickle
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode.Config.make_meta_dict_flux_tunable import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q_ft
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2R import T2R
from LabCode.Experiments.Coherence.m_T2E import T2E
from LabCode.Experiments.Coherence.m_T1_flux_ramp import T1FluxRamp
from LabCode.Experiments.Coherence.m_T2CPMG import T2CPMG
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.PythonDrivers.YOKOGS200 import YOKOGS200
from time import sleep
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

"""
Single qubit coherence measurements.
"""

# full_meta_dict = make_meta_dict_CR()
# config = meta_dict_to_config_CR(full_meta_dict)
# outer_folder = full_meta_dict["outer_folder"]
SET_YOKO_VOLTAGE = True
experiment_dict = {
    'infinite_loop': True,
    # 'meta_dict':full_meta_dict,
    # 'outer_folder': outer_folder,
    'qubits': {
        'q1': {
            "experiments": {
                'SS_Cal': {
                    'run': True,  # Must run to preform coherence experiments
                    'shots': 1000,
                    'object': SingleShot1Q,
                },
                'T1': {
                    'run': True,
                    'shots': 1000,
                    'num_pulses': [1],
                    't_min': 1 * u.us,
                    't_max': 750 * u.us,
                    't_points': 71,
                    'object': T1,
                },
                'T1_Flux_Ramp': {
                    'run': False,
                    'shots': 100,
                    'dc_excursion': 0.02,
                    't_min': 1 * u.us,
                    't_max': 750 * u.us,
                    't_points': 71,
                    'object': T1,
                },
                'T2R': {
                    'run': True,
                    'shots': 1000,
                    'detuning': -1 * u.MHz,
                    't_min': 16 * u.ns,
                    't_max': 20 * u.us,
                    't_step': 0.1 * u.us,
                    'object': T2R,
                },
                'T2E': {
                    'run': True,
                    'shots': 1000,
                    't_min': 1 * u.us,
                    't_max': 500 * u.us,
                    't_points': 71,
                    'object': T2E,
                },
                'T2CPMG': {
                    'run': False,
                    'shots': 2500,
                    't_max': [50 * u.us,] * 36,
                    't_points': 101,
                    'pi_pulses': [1, 3, 5, 7, 11,
                                  21, 31,41, 51, 61,
                                  71, 81, 91, 101, 125,
                                  151, 175, 201, 251, 301,
                                  401, 501, 601, 701, 801,
                                  901, 1001,1101, 1201, 1301,
                                  1401, 1501, 1751, 2001, 2251,
                                  2501,],
                    'object': T2CPMG,
                },
            },
        },
        'q2': {
                    "experiments": {
                        'SS_Cal': {
                            'run': True,  # Must run to preform coherence experiments
                            'shots': 1000,
                            'object': SingleShot1Q,
                        },
                        'T1': {
                            'run': True,
                            'shots': 1000,
                            'num_pulses': [1],
                            't_min': 1 * u.us,
                            't_max': 750 * u.us,
                            't_points': 71,
                            'object': T1,
                        },
                        'T2R': {
                            'run': True,
                            'shots': 1000,
                            'detuning': -0.6 * u.MHz,
                            't_min': 16 * u.ns,
                            't_max': 40 * u.us,
                            't_step': 40 * u.ns,
                            'object': T2R,
                        },
                        'T2E': {
                            'run': True,
                            'shots': 1000,
                            't_min': 1 * u.us,
                            't_max': 500 * u.us,
                            't_points': 71,
                            'object': T2E,
                        },
                        'T2CPMG': {
                            'run': False,
                            'shots': 2500,
                            't_max': [50 * u.us,] * 15 + [100*u.us]*10,
                            't_points': 101,
                            'pi_pulses': [1,3,5,7,11,
                                          21, 31, 41, 51, 75,
                                          101, 125, 151, 175, 201,
                                          251, 301, 401, 501, 601,
                                          701, 801, 901, 1001, 1201],
                            'object': T2CPMG,
                        },
                    },

        },
        'q3': {
                "experiments": {
                        'SS_Cal': {
                            'run': True,  # Must run to preform coherence experiments
                            'shots': 1000,
                            'object': SingleShot1Q,
                        },
                        'T1': {
                            'run': True,
                            'shots': 1000,
                            'num_pulses': [1],
                            't_min': 1 * u.us,
                            't_max': 1000 * u.us,
                            't_points': 71,
                            'object': T1,
                        },
                        'T2R': {
                            'run': True,
                            'shots': 1000,
                            'detuning': -1 * u.MHz,
                            't_min': 16 * u.ns,
                            't_max': 60 * u.us,
                            't_step': 100 * u.ns,
                            'object': T2R,
                        },
                        'T2E': {
                            'run': True,
                            'shots': 1000,
                            't_min': 1 * u.us,
                            't_max': 300 * u.us,
                            't_points': 71,
                            'object': T2E,
                        },
                        'T2CPMG': {
                            'run': False,
                            'shots': 2500,
                            't_max': [50 * u.us,] * 36,
                            't_points': 101,
                            'pi_pulses': [1, 3, 5, 7, 11,
                                          21, 31,41, 51, 61,
                                          71, 81, 91, 101, 125,
                                          151, 175, 201, 251, 301,
                                          401, 501, 601, 701, 801,
                                          901, 1001,1101, 1201, 1301,
                                          1401, 1501, 1751, 2001, 2251,
                                          2501,],
                            'object': T2CPMG,
                         },
                },
        },
        'q4': {
                "experiments": {
                        'SS_Cal': {
                            'run': True,  # Must run to preform coherence experiments
                            'shots': 1000,
                            'object': SingleShot1Q,
                        },
                        'T1': {
                            'run': True,
                            'shots': 1000,
                            'num_pulses': [1],
                            't_min': 1 * u.us,
                            't_max': 750 * u.us,
                            't_points': 71,
                            'object': T1,
                        },
                        'T2R': {
                            'run': True,
                            'shots': 1000,
                            'detuning': -0.4 * u.MHz,
                            't_min': 16 * u.ns,
                            't_max': 20 * u.us,
                            't_step': 60 * u.ns,
                            'object': T2R,
                        },
                        'T2E': {
                            'run': True,
                            'shots': 1000,
                            't_min': 1 * u.us,
                            't_max': 100 * u.us,
                            't_points': 71,
                            'object': T2E,
                        },
                        'T2CPMG': {
                            'run': False,
                            'shots': 2500,
                            't_max': [20*u.us]*3,
                            't_points': 101,
                            'pi_pulses': [3, 7, 15],
                            'object': T2CPMG,
                         },
                },
        },
        'q5': {
            "experiments": {
                'SS_Cal': {
                    'run': True,  # Must run to preform coherence experiments
                    'shots': 1000,
                    'object': SingleShot1Q,
                },
                'T1': {
                    'run': True,
                    'shots': 1000,
                    'num_pulses': [1],
                    't_min': 1 * u.us,
                    't_max': 750 * u.us,
                    't_points': 71,
                    'object': T1,
                },
                'T2R': {
                    'run': True,
                    'shots': 1000,
                    'detuning': -0.1 * u.MHz,
                    't_min': 16 * u.ns,
                    't_max': 45 * u.us,
                    't_step': 120 * u.ns,
                    'object': T2R,
                },
                'T2E': {
                    'run': True,
                    'shots': 1000,
                    't_min': 1 * u.us,
                    't_max': 100 * u.us,
                    't_points': 71,
                    'object': T2E,
                },
                'T2CPMG': {
                    'run': False,
                    'shots': 2500,
                    't_max': [50 * u.us, ] * 36,
                    't_points': 101,
                    'pi_pulses': [1, 3, 5, 7, 11,
                                  21, 31, 41, 51, 61,
                                  71, 81, 91, 101, 125,
                                  151, 175, 201, 251, 301,
                                  401, 501, 601, 701, 801,
                                  901, 1001, 1101, 1201, 1301,
                                  1401, 1501, 1751, 2001, 2251,
                                  2501, ],
                    'object': T2CPMG,
                },
            },
        },
        'q6': {
                    "experiments": {
                        'SS_Cal': {
                            'run': True,  # Must run to preform coherence experiments
                            'shots': 1000,
                            'object': SingleShot1Q,
                        },
                        'T1': {
                            'run': True,
                            'shots': 1000,
                            'num_pulses': [1],
                            't_min': 1 * u.us,
                            't_max': 750 * u.us,
                            't_points': 71,
                            'object': T1,
                        },
                        'T2R': {
                            'run': True,
                            'shots': 1000,
                            'detuning': -0.4 * u.MHz,
                            't_min': 1610 * u.ns,
                            't_max': 100 * u.us,
                            't_step': 60 * u.ns,
                            'object': T2R,
                        },
                        'T2E': {
                            'run': True,
                            'shots': 1000,
                            't_min': 1 * u.us,
                            't_max': 1000 * u.us,
                            't_points': 71,
                            'object': T2E,
                        },
                        'T2CPMG': {
                            'run': False,
                            'shots': 2500,
                            't_max': [30*u.us]*5 + [60*u.us]*5 + [100*u.us]*5 +
                                     [150*u.us]*10 + [250*u.us]*10,
                            't_points': 101,
                            'pi_pulses': [1, 2, 3, 4, 5,
                                          7, 9, 11, 15, 21,
                                          31, 41, 51, 61, 71,
                                          81, 91, 101, 121, 151,
                                          171, 201, 251, 301, 351,
                                          401, 501, 601, 701, 801,
                                          901, 1001, 1201, 1501, 1701],
                            'object': T2CPMG,
                         },
                     },
        },
    }
}

print("Running the Following Experiments:")
if experiment_dict['infinite_loop']:
    print("INFINITE LOOP")
    print("####################################################")

for qubit, qubit_dict in experiment_dict['qubits'].items():
    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment_params['run']:
            print(f"{qubit} {experiment}")
print("####################################################")
VISAaddress = 'USB0::0x0B21::0x0039::91S929899::0::INSTR'
yoko = YOKOGS200(VISAaddress)
ramp_step = yoko._rampstep
ramp_interval = yoko._rampinterval

while True:
    # Run Experiments
    for qubit, qubit_dict in experiment_dict['qubits'].items():
        meta_dict = make_meta_dict(qubit)[qubit]
        experiment_dict['outer_folder'] = meta_dict['global_params']['outer_folder']
        config = meta_dict_to_config_1Q_ft(meta_dict)
        LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
        print(f"{qubit} LO: {LOs}")
        # set_LO_parameters(LOs)
        if SET_YOKO_VOLTAGE:
            YOKO_VOLTAGE = meta_dict['YOKO_dc_offset']
            print(f"Setting Voltage to {YOKO_VOLTAGE} V")
            yoko.SetVoltage(YOKO_VOLTAGE)
            while yoko.GetVoltage() != YOKO_VOLTAGE:
                sleep(0.1)

        for experiment, experiment_params in qubit_dict['experiments'].items():
            if experiment == "SS_Cal" and experiment_params['run']:
                SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict,
                                  outer_folder=experiment_dict['outer_folder'], path=qubit,
                                  suffix=experiment, config=config)

                if SS.max_F < 0.6:
                    print(f"SS Cal for {qubit} failed, passing to next")
                    break
                else:
                    qubit_dict['calib_params'] = SS.calib_params
                    meta_dict['read_theta'] = SS.calib_params["read_theta"]
                    config = meta_dict_to_config_1Q_ft(meta_dict)
                    print(f"SS Cal for {qubit} complete")

            if experiment == "T1" and experiment_params['run']:
                t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                    experiment_params['t_points'])
                if experiment_params['num_pulses']:
                    for num_pulse in experiment_params['num_pulses']:
                        exp = T1(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                                 meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                 path=qubit, suffix=experiment, config=config, num_pulses=num_pulse)
                else:
                    exp = T1(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                             meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                             path=qubit, suffix=experiment, config=config)
                print(f"T1 {qubit} complete")
            if experiment == "T1_Flux_Ramp" and experiment_params['run']:
                t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                    experiment_params['t_points'])
                exp = T1FluxRamp(qubit, t_vec, experiment_params['dc_excursion'], experiment_params['shots'],  calib_params=qubit_dict['calib_params'],
                         meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                         path=qubit, suffix=experiment, config=config)
                print(f"T1 {qubit} complete")

            if experiment == "T2R" and experiment_params['run']:
                t_vec = np.arange(experiment_params['t_min'], experiment_params['t_max'], experiment_params['t_step'])
                exp = T2R(qubit, t_vec, experiment_params['shots'], experiment_params['detuning'],
                          calib_params=qubit_dict['calib_params'], meta_dict=meta_dict,
                          outer_folder=experiment_dict['outer_folder'],
                          path=qubit, suffix=experiment, config=config)
                print(f"T2R {qubit} complete")

            if experiment == "T2E" and experiment_params['run']:
                t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                    experiment_params['t_points'])
                exp = T2E(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                          meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                          path=qubit, suffix=experiment, config=config)
                print(f"T2E {qubit} complete")

            if experiment == "T2CPMG" and experiment_params['run']:
                assert len(experiment_params['pi_pulses']) == len(experiment_params['t_max'])
                data = []
                error = []
                pulses = []
                for i, pi_pulses in enumerate(experiment_params['pi_pulses']):
                    t_min = (16 + pi_pulses * 20) * u.ns
                    if t_min < 500 * u.ns:
                        t_min = 500*u.ns

                    print(pi_pulses)
                    t_max = experiment_params['t_max'][i]
                    t_vec = np.linspace(t_min, t_max, experiment_params['t_points'])
                    # t_vec = np.logspace(np.log10(t_min), np.log10(t_max), experiment_params['t_points'])
                    exp = T2CPMG(qubit, t_vec, pi_pulses, experiment_params['shots'],
                                 calib_params=qubit_dict['calib_params'], meta_dict=meta_dict,
                                 outer_folder=experiment_dict['outer_folder'],
                                 path=qubit, suffix=experiment, config=config)
                    # try:
                    #     T2, err = exp.data['T2CPMG']
                    # except:
                    #     T2 = 0
                    #     err = 0
                    # data.append(T2)
                    # error.append(err)
                    # pulses.append(pi_pulses)
                    # output = {"T2": data,
                    #           "error": error,
                    #           'pulses': pulses
                    #           }
                    # pickle_name = exp.pname[:-12] + "_CPMG_TOTAL.pkl"
                    # with open(pickle_name, 'wb') as f:
                    #     pickle.dump(output, f)
                    #
                    # plt.errorbar(pulses, data, yerr=error)
                    # plt.title(f'{qubit} T2CPMG Convergence')
                    # plt.xlabel("Pi Pulses")
                    # plt.ylabel('T2 (us)')
                    # image_name = exp.iname[:-12] + "_CPMG_TOTAL.png"
                    # # plt.savefig(image_name, bbox_inches="tight")
                    # plt.close()
                print(f"T2CPMG {qubit} complete")
    if not experiment_dict['infinite_loop']:
        break


