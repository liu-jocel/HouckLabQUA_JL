import matplotlib.pyplot as plt
from time import sleep
import h5py
import numpy as np
import matplotlib
import copy
matplotlib.use('TkAgg')
from LabCode.PythonDrivers.YOKOGS200 import YOKOGS200
from LabCode.Config.make_meta_dict_flux_tunable import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q_ft
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.m_transmission import Transmission
from LabCode.Experiments.m_chi import ChiMeasurement
from LabCode.Experiments.m_transmission_sweep import TransmissionSweep
from LabCode.Experiments.m_qubit_spec import QubitSpec
from LabCode.Experiments.Flux_Sweeps.m_qubit_spec_vs_flux import QubitSpecVsFlux
from LabCode.Experiments.Flux_Sweeps.m_transmission_vs_flux import TransmissionVsFlux
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses.
"""

YOKO_VOLTAGE = 0
experiment_dict = {
    'qubits': {
        'q1': {
            "experiments": {
                'Transmission_Sweep':{
                    'run': True,
                    'shots': 200,
                    'a_min': 0.01,
                    'a_max': 0.4,
                    'num_a': 11,
                    'freq_max': -199 * u.MHz,
                    'freq_min': -201 * u.MHz,
                    'freq_step': 0.03 * u.MHz,
                    'object': TransmissionSweep,
                },
                'Transmission': {
                    'run': False,
                    'shots': 50,
                    'freq_max': -199 * u.MHz,
                    'freq_min': -202 * u.MHz,
                    'freq_step': 0.1 * u.MHz,
                    'object': Transmission,
                },
                'Qubit_Spec': {
                    'run': False,
                    'shots': 100,
                    'spec_amp': 0.1,
                    'spec_len': 500 * u.ns,
                    'freq_max': -20 * u.MHz,
                    'freq_min': -400 * u.MHz,
                    'freq_step': 1 * u.MHz,
                    'object': QubitSpec,
                },
                'Transmission_vs_Flux': {
                    'run': False,
                    'shots': 100,
                    'freq_max': -199 * u.MHz,
                    'freq_min': -202 * u.MHz,
                    'freq_step': 0.05 * u.MHz,
                    'dc_max': 0.4,
                    'dc_min': -0.4,
                    'dc_step': 0.01,
                    'object': TransmissionVsFlux,
                },
                'Transmission_vs_Flux_YOKO': {
                    'run': False,
                    'shots': 100,
                    'freq_max': -199 * u.MHz,
                    'freq_min': -202 * u.MHz,
                    'freq_step': 0.02 * u.MHz,
                    'dc_max': 0.2,  # +- 30 V
                    'dc_min': -.2,
                    'dc_step': 0.05,
                    'object': Transmission,
                },
                'Qubit_Spec_vs_Flux': {
                    'run': False,
                    'shots': 100,  # 100,
                    'spec_amp': 0.08,  # 0.1,
                    'spec_len': 100 * u.ns,  # 100 ns
                    'freq_max': -0 * u.MHz,
                    'freq_min': -400 * u.MHz,
                    'freq_step': 1 * u.MHz,
                    'dc_max':  0.05,
                    'dc_min':  -0.03,
                    'dc_step': 0.001,
                    'object': QubitSpecVsFlux,
                },
                'Qubit_Spec_vs_Flux_YOKO': {
                    'run': False,
                    'shots': 100,  # 100,
                    'spec_amp': 0.08,  # 0.1,
                    'spec_len': 100 * u.ns,  # 100 ns
                    'freq_max': -100 * u.MHz,
                    'freq_min': -300 * u.MHz,
                    'freq_step': 0.5 * u.MHz,
                    'dc_max': 0.26,
                    'dc_min': 0.24,
                    'dc_step': 0.002,
                    'object': QubitSpecVsFlux,
                },
            },
        },
    },
}


VISAaddress = 'USB0::0x0B21::0x0039::91S929899::0::INSTR'
yoko = YOKOGS200(VISAaddress)
ramp_step = yoko._rampstep
ramp_interval = yoko._rampinterval

# yoko.SetVoltage(YOKO_VOLTAGE)
# while yoko.GetVoltage() != YOKO_VOLTAGE:
#     sleep(0.1)

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
            f_vec = np.arange(experiment_params['freq_min'], experiment_params['freq_max'], experiment_params['freq_step'])
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
        if experiment == "Transmission_vs_Flux" and experiment_params['run']:
            f_vec = np.arange(experiment_params['freq_min'], experiment_params['freq_max'], experiment_params['freq_step'])
            dc_vec = np.arange(experiment_params['dc_min'], experiment_params['dc_max'],
                                 experiment_params['dc_step'])
            exp = TransmissionVsFlux(qubit, f_vec, dc_vec, experiment_params['shots'],
                                  meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                  path=qubit, suffix=experiment, config=config)
            print(f"Spec for {qubit} complete")
        if experiment == "Transmission_vs_Flux_YOKO" and experiment_params['run']:
            f_vec = np.arange(experiment_params['freq_min'], experiment_params['freq_max'], experiment_params['freq_step'])
            dc_vec = np.arange(experiment_params['dc_min'], experiment_params['dc_max'],
                                 experiment_params['dc_step'])
            transmission_data = []
            save_paths = None
            dV = np.abs(dc_vec[1] - dc_vec[0])
            for i, V in enumerate(dc_vec):
                print(f"{i/len(dc_vec)}%", end='\r')
                print(f"Setting Voltage to {V} V")
                yoko.SetVoltage(V)
                sleep(ramp_interval * dV / ramp_step)
                print('Running Transmission')
                exp = Transmission(qubit, f_vec, experiment_params['shots'],
                                      meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                      path=qubit, suffix=experiment, config=config, plot=False, save=False)
                transmission_data.append(exp.data["IQ_mag"])
                save_paths = (exp.iname, exp.pname[:-4]+".h5")
            transmission_data = np.array((transmission_data))
            f = h5py.File(save_paths[1], 'w')
            dset = f.create_dataset('dc_vec', data=dc_vec)
            dset = f.create_dataset('IQ_mag', data=transmission_data)
            dset = f.create_dataset('f_vec', data=f_vec)
            f.close()
            X, Y = np.meshgrid(dc_vec, f_vec)
            plt.pcolor(X, Y, transmission_data.T, shading='nearest')
            plt.xlabel('Yoko Voltage (V)')
            plt.ylabel('Resonator IF (MHz)')
            plt.title("Transmission vs Flux")
            plt.colorbar()
            plt.savefig(save_paths[0])
            plt.close()
            yoko.SetVoltage(0)
            sleep(60)

            print(f"Spec for {qubit} complete")
        if experiment == "Qubit_Spec_vs_Flux" and experiment_params['run']:
            set_LO_parameters([meta_dict['q_LO']])
            MD = copy.deepcopy(meta_dict)
            MD['cw_amp'] = experiment_params['spec_amp']
            MD['cw_len'] = experiment_params['spec_len']
            config_copy = meta_dict_to_config_1Q_ft(MD)
            f_vec = np.arange(experiment_params['freq_min'], experiment_params['freq_max'], experiment_params['freq_step'])
            dc_vec = np.arange(experiment_params['dc_min'],  experiment_params['dc_max'],  experiment_params['dc_step'])
            exp = QubitSpecVsFlux(qubit, f_vec, dc_vec, experiment_params['shots'],
                            meta_dict=MD, outer_folder=experiment_dict['outer_folder'],
                            path=qubit, suffix=experiment, config=config_copy)
            print(f"Spec for {qubit} complete")
        if experiment == "Qubit_Spec_vs_Flux_YOKO" and experiment_params['run']:
            set_LO_parameters([meta_dict['q_LO']])
            f_vec = np.arange(experiment_params['freq_min'], experiment_params['freq_max'], experiment_params['freq_step'])
            dc_vec = np.arange(experiment_params['dc_min'], experiment_params['dc_max'],
                                 experiment_params['dc_step'])
            print(dc_vec)
            transmission_data = []
            save_paths = None
            for i, V in enumerate(dc_vec):
                print(f"{i/len(dc_vec)}%")
                print(f"Setting Voltage to {V} V")
                yoko.SetVoltage(V)
                while round(yoko.GetVoltage(), 4) != round(V, 4):
                    sleep(0.1)
                print('Running Transmission')
                MD = copy.deepcopy(meta_dict)
                MD['cw_amp'] = experiment_params['spec_amp']
                MD['cw_len'] = experiment_params['spec_len']
                config_copy = meta_dict_to_config_1Q_ft(MD)
                exp = QubitSpec(qubit, experiment_params['freq_min'], experiment_params['freq_max'],
                                experiment_params['freq_step'], experiment_params['shots'],
                                meta_dict=MD, outer_folder=experiment_dict['outer_folder'],
                                path=qubit, suffix=experiment, config=config_copy, plot=False, save=False)
                transmission_data.append(exp.data["IQ_magnitude_dBm"])
                save_paths = (exp.iname, exp.pname[:-4]+".h5")
            transmission_data = np.array((transmission_data))
            f = h5py.File(save_paths[1], 'w')
            dset = f.create_dataset('dc_vec', data=dc_vec)
            dset = f.create_dataset('IQ_mag', data=transmission_data)
            dset = f.create_dataset('f_vec', data=f_vec)
            f.close()
            X, Y = np.meshgrid(dc_vec, f_vec)
            plt.pcolor(X, Y, transmission_data.T, shading='nearest')
            plt.xlabel('Yoko Voltage (V)')
            plt.ylabel('Qubit IF (MHz)')
            plt.title("Qubit Spec vs Flux")
            plt.colorbar()
            plt.savefig(save_paths[0])
            plt.close()
            yoko.SetVoltage(0)
            sleep(10)