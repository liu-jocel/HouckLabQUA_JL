import numpy as np
import matplotlib.pyplot as plt
from LabCode.Config.make_meta_dict_flux_tunable import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q_ft
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_SS_Cal_1Q import SS_Cal_1Q
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


"""
File to control readout calibration.
"""
# Get metaDict
# full_meta_dict = make_meta_dict_CR()
# outer_folder = full_meta_dict['outer_folder']
# config = meta_dict_to_config_CR(full_meta_dict)

experiment_dict = dict(
    # meta_dict=full_meta_dict,
    # outer_folder=outer_folder,
    qubits={
        'q4': {
            "experiments": {
                'SS_Cal': {
                    'run': True,  # Must run to preform SS experiments
                    'shots': 10000,
                    'number_pi_pulses': 1,
                    'object': SingleShot1Q,
                },
                'Readout_Cal_1Q': {
                    'run': True,
                    'shots': 100,
                    'f_span': 1 * u.MHz,
                    'num_f': 7,
                    'a_min': 0.01,  # V
                    'a_max': 0.06,  # V. Note: must be < 2 * read_amp
                    'num_a': 9,
                    'number_pi_pulses': 1,
                    'object': SS_Cal_1Q,
                },
            },
        },
    },
)
print("Running the Following Experiments:")
print("####################################################")
for qubit, qubit_dict in experiment_dict['qubits'].items():
    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment_params['run']:
            print(f"{qubit} {experiment}")
print("####################################################")

# Run Experiments
for qubit, qubit_dict in experiment_dict['qubits'].items():
    # meta_dict = experiment_dict["meta_dict"][qubit]
    # # meta_dict['global_params'] = full_meta_dict['global_params']
    # config = meta_dict_to_config_CR(full_meta_dict)
    # LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
   #  # set_LO_parameters(LOs)
    meta_dict = make_meta_dict(qubit)[qubit]

    experiment_dict['outer_folder'] = meta_dict['global_params']['outer_folder']
    config = meta_dict_to_config_1Q_ft(meta_dict)
    # config = meta_dict_to_config_CR(meta_dict)
    LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
    print(LOs)
    # set_LO_parameters(LOs)

    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment == "SS_Cal" and experiment_params['run']:
            SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict, config=config,
                                outer_folder=experiment_dict['outer_folder'], path=qubit,
                                suffix=experiment)

        if experiment == "Readout_Cal_1Q" and experiment_params['run']:
            r_IF = meta_dict['r_IF']
            amp_vec = np.linspace(experiment_params['a_min'], experiment_params['a_max'], experiment_params['num_a'])
            f_vec = np.linspace(-experiment_params['f_span']/2, experiment_params['f_span']/2, experiment_params['num_f'])
            print(f'Readout_Cal_1Q est. completion time: {len(amp_vec)*len(f_vec)*experiment_params["shots"]*meta_dict["global_params"]["reset_time"]/u.s /60} Minutes')

            fidels = np.zeros([len(f_vec), len(amp_vec)])
            for j, amp in enumerate(amp_vec):
                meta_dict['read_amp'] = amp
                for i, df in enumerate(f_vec):
                    meta_dict['r_IF'] = r_IF + df
                    config = meta_dict_to_config_1Q_ft(meta_dict)
                    SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict, config=config,
                                outer_folder=experiment_dict['outer_folder'], path=qubit,
                                suffix=experiment, plot=False, save=False, confidence_threshold=0)
                    if i == 0:
                        if j == 0:
                            name = SS.iname[:-6] + '2Q'

                    fidels[i][j] = SS.max_F
                    if SS.max_F > 0.85:
                        print(SS.max_F)
                        print(amp)
                        print((df + r_IF)/u.MHz)

                    vmin = np.max([np.min([fidels]), 0.5])
                    vmax = np.max([fidels])

                    f_idx_max, a_idx_max = np.unravel_index(np.argmax(fidels), fidels.shape)

                    fig, ax = plt.subplots(1, 1, constrained_layout=True)
                    pcm1 = ax.pcolormesh(amp_vec, (f_vec + r_IF)/u.MHz, fidels, vmin=vmin, vmax=vmax)
                    ax.set_xlabel('Readout amplitude (V)')
                    ax.set_ylabel('Resonator IF (MHz)')
                    ax.set_title(f"Readout Cal {qubit}, optimal freq: {(f_vec[f_idx_max]+r_IF)/u.MHz} MHz. amp optimal: {amp_vec[a_idx_max]} ")
                    fig.colorbar(pcm1, ax=ax)
                    plt.savefig(name + '2QC', bbox_inches='tight')
                    plt.close()

            print(f"Optimal Freq: {(f_vec[f_idx_max]+r_IF)/u.MHz} MHz")
            print(f"Optimal amp: {amp_vec[a_idx_max]}")
