import numpy as np
import matplotlib
import copy
# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_6Q
from LabCode.Config.config_helpers import meta_dict_to_config_6Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Gates.m_ZZ_crosstalk import ZZCrosstalk
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

TEST_QUBIT = "q4"
CROSSTALK_QUBIT = "q6"
experiment_dict = {
            'SS_Cal': {
                'shots': 300,
                'number_pi_pulses': 1,
                'ground_threshold': 0.7,
                'object': SingleShot1Q,
            },
            'ZZ_Crosstalk': {
                    'run': True,
                    'shots': 100,
                    'detuning': 0.5 * u.MHz,
                    't_min': 16 * u.ns,
                    't_max': 10 * u.us,
                    't_step': 32 * u.ns,
                    'object': ZZCrosstalk,
                },
}


#Set LO for Expertiment
meta_dict = make_meta_dict_6Q()
config = meta_dict_to_config_6Q(meta_dict)
LOs = [meta_dict[TEST_QUBIT][TEST_QUBIT]['q_LO'], meta_dict[TEST_QUBIT][TEST_QUBIT]['r_LO']]
# set_LO_parameters(LOs)

# Calibrate Readout
SS = SingleShot1Q(TEST_QUBIT, experiment_dict["SS_Cal"]['shots'], meta_dict=meta_dict[TEST_QUBIT][TEST_QUBIT],
                  outer_folder=meta_dict[TEST_QUBIT]['global_params']['outer_folder'], path=TEST_QUBIT,
                  suffix="SS_Cal", repeats=experiment_dict["SS_Cal"]['number_pi_pulses'],
                  config=config, confidence_threshold=experiment_dict["SS_Cal"]['ground_threshold'])
CALIB_PARAMS = SS.calib_params
fidelity = SS.max_F
meta_dict[TEST_QUBIT][TEST_QUBIT]['read_theta'] = SS.calib_params["read_theta"]
config = meta_dict_to_config_6Q(meta_dict)

t_vec = np.arange(experiment_dict['ZZ_Crosstalk']['t_min'], experiment_dict['ZZ_Crosstalk']['t_max'], experiment_dict['ZZ_Crosstalk']['t_step'])
exp = ZZCrosstalk(TEST_QUBIT, CROSSTALK_QUBIT, t_vec, experiment_dict['ZZ_Crosstalk']['shots'], experiment_dict['ZZ_Crosstalk']['detuning'],
                      calib_params=CALIB_PARAMS, meta_dict=meta_dict,
                      outer_folder=meta_dict[TEST_QUBIT]['global_params']['outer_folder'],
                      path=TEST_QUBIT, suffix="ZZ_Crosstalk", config=config)


