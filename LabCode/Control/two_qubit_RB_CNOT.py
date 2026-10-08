import matplotlib.pyplot as plt
import numpy as np
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_2Q import SingleShot2Q
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_CR
from LabCode.Config.config_helpers import meta_dict_to_config_CR
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


meta_dict = make_meta_dict_CR()
outer_folder = meta_dict["outer_folder"]
config = meta_dict_to_config_CR(meta_dict)

# Make sure LO's for each qubit are the same!!
LOs = [meta_dict['q1']['q_LO'], meta_dict['q1']['r_LO']]
# set_LO_parameters(LOs)

# Make sure LO's for each qubit are the same!!
LOs = [meta_dict['q1']['q_LO'], meta_dict['q1']['r_LO']]
# set_LO_parameters(LOs)


experiment_dict = {
    'Infinite_Loop': False,
    'meta_dict': meta_dict,
    'outer_folder': outer_folder,
    "experiments": {
        "SS_Cal_2Q":{
            'run': True,  # Must run to preform SS experiments
            'shots': 1000,
            'ground_state_threshold': 0.7,
            'object': SingleShot2Q,
        },
        "2QRB_CNOT": {
            'run': False,
            'shots': 100,
            'num_circuits': 20,
            'circuit_lenghts': [2, 16, 64, 128, 256, 512, 750, 1000, 1250, 1500, 1750, 2000,],
            'simulate': False
        },
    },
},


print("Running the Following Experiments:")
print("####################################################")
for experiment, experiment_params in experiment_dict['experiments'].items():
    if experiment_params['run']:
        print(f"{experiment}")
print("####################################################")

PATH = "2QRB"
for experiment, experiment_params in experiment_dict['experiments'].items():
    if experiment == "SS_Cal_2Q" and experiment_params['run']:
        SS = None
        trace = 0
        while trace/4 < 0.82:
            SS = SingleShot2Q("q1", "q2", experiment_params['shots'], meta_dict=meta_dict,
                              outer_folder=experiment_dict['outer_folder'], path=PATH,
                              suffix=experiment, config=config, confidence_threshold=experiment_params['ground_state_threshold'])
            print(SS.confusion)
            trace = np.trace(np.array(SS.confusion))
            print(trace/4)
        SS.pickle_data()
        experiment_dict['calib_params'] = SS.calib_params
        experiment_dict['confusion_matrix'] = SS.confusion
        meta_dict["q1"]['read_theta'] = SS.calib_params["read_theta_q1"]
        meta_dict["q2"]['read_theta'] = SS.calib_params["read_theta_q2"]
        config = meta_dict_to_config_CR(meta_dict)
        print("SS Cal Complete")

    if experiment == "2QRB_CNOT" and experiment_params['run']:
        total_cycles = len(experiment_params['circuit_lenghts']) * experiment_params['num_circuits'] * \
                       experiment_params['shots']
        exp = RB2Q_CNOT(experiment_params['circuit_lenghts'], experiment_params['num_circuits'],
                   experiment_params['shots'], experiment_dict['calib_params'], experiment_dict['confusion_matrix'],
                   meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                   path=PATH, suffix=experiment + "_all90", config=config)