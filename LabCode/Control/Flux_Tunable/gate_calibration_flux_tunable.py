import numpy as np
import copy
from time import sleep
from LabCode.PythonDrivers.YOKOGS200 import YOKOGS200
from LabCode.Config.make_meta_dict_flux_tunable import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q_ft
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.m_transmission import Transmission
from LabCode.Experiments.m_chi import ChiMeasurement
from LabCode.Experiments.m_transmission_sweep import TransmissionSweep
from LabCode.Experiments.m_qubit_spec import QubitSpec
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Rabi.m_Rabi_Linecut_SS import RabiLinecutSS
from LabCode.Experiments.Rabi.m_Rabi_Chevron_Timed_IQ import RabiChevronTimedIQ
from LabCode.Experiments.Rabi.m_Rabi_Linecut_Timed_IQ import RabiLinecutTimedIQ
from LabCode.Experiments.Rabi.m_freq_cal import FreqCal1Q
from LabCode.Experiments.DRAG.m_DRAG_cal import DragCal1Q
import matplotlib
matplotlib.use('qtagg', force=True)
import matplotlib.pyplot as plt
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses.
"""

YOKO_VOLTAGE = -0.01
PULSE_TYPE = "X90"
experiment_dict = {
    'qubits':{
        'q5': {
            "experiments": {
                'Transmission_Sweep':{
                    'run': False,
                    'shots': 200,
                    'a_min': 0.01,
                    'a_max': 0.05,
                    'num_a': 5,
                    'freq_min': -202 * u.MHz,
                    'freq_max': -198 * u.MHz,
                    'freq_step': 0.05 * u.MHz,
                    'object': TransmissionSweep,
                },
                'Transmission': {
                    'run': False,
                    'shots': 200,
                    'freq_min': -202 * u.MHz,
                    'freq_max': -198 * u.MHz,
                    'freq_step': 0.05 * u.MHz,
                    'object': Transmission,
                },
                'Chi_Measurement': {
                    'run': False,
                    'shots': 50,
                    'freq_min': -202 * u.MHz,
                    'freq_max': -199 * u.MHz,
                    'freq_step': 0.05 * u.MHz,
                    'object': ChiMeasurement,
                },
                'Qubit_Spec': {
                    'run': False,
                    'shots': 100,
                    'spec_amp': 0.01,  # 0.1,
                    'spec_len': 100 * u.ns,  # 100 ns
                    'freq_max': -20 * u.MHz,
                    'freq_min': -400 * u.MHz,
                    'freq_step': 1 * u.MHz,
                    'object': QubitSpec,
                },
                'SS_Cal': {
                    'run': True,  # Must run to preform SS experiments
                    'shots': 1000,
                    'number_pi_pulses': 1,
                    'ground_threshold': 0.7,
                    'object': SingleShot1Q,
                },
                'Rabi_Chevron_IQ': {
                    'run': True,
                    'shots': 50,
                    'number_pi_pulses': 3,
                    'a_min': 0.03,  # Volts
                    'a_max': 0.06,  # Volts note: 2 * pulse amplitude
                    'a_points': 21,
                    'freq_span': 11 * u.MHz,
                    'freq_points': 21,
                    'pulse_type': PULSE_TYPE,
                    'object': RabiChevronIQ,
                },

                'Rabi_Chevron_SS': {
                    'run': False,
                    'shots': 300,
                    'number_pi_pulses': 3,
                    'a_min': 0.02,  # Volts
                    'a_max': 0.04,  # Volts note: 2 * pulse amplitude
                    'a_points': 21,
                    'freq_span': 11 * u.MHz,
                    'freq_points': 21,
                    'pulse_type': PULSE_TYPE,
                    'object': RabiChevronSS,
                },
                'Rabi_Linecut_SS': {
                    'run': False,
                    'shots': 1000,
                    'number_pi_pulses': 641,
                    'a_span': 0.0002,  #,  # Volts. Note: must be less than  2 * pulse amplitude
                    'a_points': 51,
                    'pulse_type': PULSE_TYPE,
                    'object': RabiLinecutSS,
                },
                'Frequency_Calibration': {
                    'run': False,
                    'shots': 1000,
                    'number_pi_pulses': 81,  # experiment does X80 followed by mX180
                    'freq_span': 0.4 * u.MHz,
                    'freq_points': 51,
                    'pulse_type': PULSE_TYPE,
                    'object': FreqCal1Q,
                },
                'DRAG_Calibation': {
                    'run': False,
                    'shots': 200,
                    'number_pi_pulses': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],   # experiment does X80 followed by mX180
                    'alpha_min': -2,  # Volts
                    'alpha_max': 2,  # Volts. Note: must be less than  2 * pulse amplitude
                    'alpha_step': 0.05,
                    'pulse_type': PULSE_TYPE,
                    'object': DragCal1Q,
                },
                'Rabi_Timed_Chevron_IQ': {
                    'run': False,
                    'shots': 300,
                    'number_pi_pulses': 1,
                    'pi_amp': 0.,
                    't_max': 100 * u.ns,
                    't_step': 4 * u.ns,
                    'freq_span': 21 * u.MHz,
                    'freq_points': 101,
                    'pulse_type': PULSE_TYPE,
                    'object': RabiChevronTimedIQ,
                },
                'Rabi_Timed_Linecut_IQ': {
                    'run': False,
                    'shots': 1000,
                    'number_pi_pulses': 1,
                    'pi_amp': 0.08,
                    't_max': 200 * u.ns,
                    't_step': 4 * u.ns,
                    'pulse_type': PULSE_TYPE,
                    'object': RabiLinecutTimedIQ,
                },
            },
        },
    },
}



VISAaddress = 'USB0::0x0B21::0x0039::91S929899::0::INSTR'
yoko = YOKOGS200(VISAaddress)
ramp_step = yoko._rampstep
ramp_interval = yoko._rampinterval

yoko.SetVoltage(YOKO_VOLTAGE)
while yoko.GetVoltage() != YOKO_VOLTAGE:
    sleep(0.1)

print("Running the Following Experiments:")
print("####################################################")
for qubit, qubit_dict in experiment_dict['qubits'].items():
    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment_params['run']:
            print(f"{qubit} {experiment}")
print("####################################################")

for qubit, qubit_dict in experiment_dict['qubits'].items():

    meta_dict = make_meta_dict(qubit)[qubit]

    experiment_dict['outer_folder'] = meta_dict['global_params']['outer_folder']
    config = meta_dict_to_config_1Q_ft(meta_dict)
    LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
    print(LOs)
    # set_LO_parameters(LOs)

    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment == "Transmission_Sweep" and experiment_params['run']:
            copy_MD = copy.deepcopy(meta_dict)
            copy_MD['read_amp'] = 0.45
            copy_config = meta_dict_to_config_1Q_ft(copy_MD)
            f_vec = np.arange(experiment_params['freq_min'],
                              experiment_params['freq_max'], experiment_params['freq_step'])
            a_vec = np.linspace(experiment_params['a_min'],
                              experiment_params['a_max'], experiment_params['num_a'])
            exp = TransmissionSweep(qubit,a_vec, f_vec, experiment_params['shots'],
                               meta_dict=copy_MD, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=copy_config)
            print(f"Transmission for {qubit} complete")
        if experiment == "Transmission" and experiment_params['run']:
            f_vec = np.arange(experiment_params['freq_min'],
                              experiment_params['freq_max'], experiment_params['freq_step'])
            exp = Transmission(qubit, f_vec, experiment_params['shots'],
                               meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=config)
            print(f"Transmission Sweep for {qubit} complete")
        if experiment == "Chi_Measurement" and experiment_params['run']:
            exp = ChiMeasurement(qubit, experiment_params['freq_min'], experiment_params['freq_max'],
                               experiment_params['freq_step'], experiment_params['shots'],
                               meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=config)
            print(f"Transmission for {qubit} complete")
        if experiment == "Qubit_Spec" and experiment_params['run']:
            set_LO_parameters([meta_dict['q_LO']])
            MD = copy.deepcopy(meta_dict)
            MD['cw_amp'] = experiment_params['spec_amp']
            MD['cw_len'] = experiment_params['spec_len']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            exp = QubitSpec(qubit, experiment_params['freq_min'], experiment_params['freq_max'],
                            experiment_params['freq_step'], experiment_params['shots'],
                            meta_dict=MD, outer_folder=experiment_dict['outer_folder'],
                            path=qubit, suffix=experiment, config=config_copy)
            print(f"Spec for {qubit} complete")
        if experiment == "SS_Cal" and experiment_params['run']:
            fidelity = 0
            while fidelity < 0.1:
                SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict,
                                  outer_folder=experiment_dict['outer_folder'], path=qubit,
                                    suffix=experiment, repeats=experiment_params['number_pi_pulses'],
                                  config=config, confidence_threshold=experiment_params['ground_threshold'])
                qubit_dict['calib_params'] = SS.calib_params
                fidelity = SS.max_F
                meta_dict['read_theta'] = SS.calib_params["read_theta"]
                config = meta_dict_to_config_1Q_ft(meta_dict)
                print(f"SS Cal for {qubit} complete")
        if experiment == "Rabi_Chevron_IQ" and experiment_params['run']:
            MD = copy.deepcopy(meta_dict)
            MD['pi_amp'] = experiment_params['a_max']
            MD['pi2_amp'] = experiment_params['a_max']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            amp_vec = np.linspace(experiment_params['a_min'], experiment_params['a_max'], experiment_params['a_points'])
            freq_vec = np.linspace(-experiment_params['freq_span']/2, experiment_params['freq_span']/2,
                                   experiment_params['freq_points'], dtype=int)
            exp = RabiChevronIQ(qubit, amp_vec,freq_vec, experiment_params['shots'],
                            experiment_params['number_pi_pulses'], pulse_type=experiment_params['pulse_type'], meta_dict=MD,
                            outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment, config=config_copy)
            print(f"Rabi Chevron IQ for {qubit} complete")
        if experiment == "Rabi_Timed_Chevron_IQ" and experiment_params['run']:
            MD = copy.deepcopy(meta_dict)
            MD['pi_amp'] = experiment_params['pi_amp']
            MD['pi2_amp'] = experiment_params['pi_amp']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            t_vec = np.arange(4*u.ns, experiment_params['t_max'], experiment_params['t_step'])
            freq_vec = np.linspace(-experiment_params['freq_span']/2, experiment_params['freq_span']/2,
                                   experiment_params['freq_points'], dtype=int)
            exp = RabiChevronTimedIQ(qubit, t_vec,freq_vec, experiment_params['shots'], meta_dict=MD,
                            outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment, config=config_copy)
            print(f"Rabi Chevron IQ Timed for {qubit} complete")
        if experiment == "Rabi_Timed_Linecut_IQ" and experiment_params['run']:
            MD = copy.deepcopy(meta_dict)
            MD['pi_amp'] = experiment_params['pi_amp']
            MD['pi2_amp'] = experiment_params['pi_amp']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            t_vec = np.arange(4*u.ns, experiment_params['t_max'], experiment_params['t_step'])
            exp = RabiLinecutTimedIQ(qubit, t_vec, experiment_params['shots'], meta_dict=MD,
                            outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment, config=config_copy)
            print(f"Rabi Chevron IQ Timed for {qubit} complete")
        if experiment == "Rabi_Chevron_SS" and experiment_params['run']:
            MD = copy.deepcopy(meta_dict)
            MD['pi_amp'] = experiment_params['a_max']
            MD['pi2_amp'] = experiment_params['a_max']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            amp_vec = np.linspace(experiment_params['a_min'], experiment_params['a_max'], experiment_params['a_points'])
            freq_vec = np.linspace(-experiment_params['freq_span']/2, experiment_params['freq_span']/2,
                                   experiment_params['freq_points'], dtype=int)
            exp = RabiChevronSS(qubit, amp_vec,freq_vec, experiment_params['shots'], qubit_dict['calib_params'],
                            experiment_params['number_pi_pulses'], pulse_type=experiment_params['pulse_type'], meta_dict=MD,
                            outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment, config=config_copy)
            print(f"Rabi_Chevron_SS for {qubit} complete")
        if experiment == "Rabi_Linecut_SS" and experiment_params['run']:
            if experiment_params['pulse_type'] == 'X180':
                amp = meta_dict['pi_amp']
            if experiment_params['pulse_type'] == 'X90':
                amp = meta_dict['pi2_amp']
            amp_vec = np.linspace(amp-experiment_params['a_span']/2, amp+experiment_params['a_span']/2, experiment_params['a_points'])
            if type(experiment_params['number_pi_pulses']) == list:
                data = []
                for i, num_pi in enumerate(experiment_params['number_pi_pulses']):
                    exp = RabiLinecutSS(qubit, amp_vec, experiment_params['shots'], qubit_dict['calib_params'],
                                        num_pi_pulses=num_pi, pulse_type=experiment_params['pulse_type'],
                                        meta_dict=meta_dict,
                                        outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment,
                                        config=config)
                    data.append(exp.data['ss_data']),
                plt.pcolor(data, shading='auto')
                i_name = exp.iname[:-4]+"_pi_sweep.png"
                print(i_name)
                plt.savefig(i_name)
            else:
                exp = RabiLinecutSS(qubit, amp_vec, experiment_params['shots'], qubit_dict['calib_params'],
                                    num_pi_pulses=experiment_params['number_pi_pulses'], pulse_type=experiment_params['pulse_type'],
                                    meta_dict=meta_dict,
                                    outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment,
                                    config=config)
            print(f"Rabi_Linecut_SS for {qubit} complete")
        if experiment == "Frequency_Calibration" and experiment_params['run']:
            freq_vec = np.linspace(-experiment_params['freq_span'] / 2, experiment_params['freq_span'] / 2,
                                   experiment_params['freq_points'], dtype=int)
            exp = FreqCal1Q(qubit, freq_vec, experiment_params['shots'], qubit_dict['calib_params'],
                            experiment_params['number_pi_pulses'], pulse_type=experiment_params['pulse_type'],
                            meta_dict=meta_dict,
                            outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment, config=config)
            print(f"Frequency_Calibration for {qubit} complete")
        if experiment == 'DRAG_Calibation' and experiment_params['run']:
            # alpha_vec = np.arange(experiment_params['alpha_min'], experiment_params['alpha_max'], experiment_params['alpha_step'])
            # full_meta_dict[qubit]['DRAG_beta'] = 1
            # meta_dict = full_meta_dict[qubit]
            # config = meta_dict_to_config_CR(full_meta_dict)
            # # print(f'DRAG_Calibation expected time: {len(alpha_vec) * len(experiment_params["number_pi_pulses"]) * experiment_params["shots"] * full_meta_dict["global_params"]["reset_time"] / u.s / 60} Minutes')
            # exp = DragCal1Q(qubit, alpha_vec, experiment_params['number_pi_pulses'], experiment_params['shots'],
            #              qubit_dict['calib_params'], pulse_type=experiment_params['pulse_type'], meta_dict=meta_dict, config=config,
            #              outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment)
            # print(f"DRAG_Calibation for {qubit} complete")
            pass
