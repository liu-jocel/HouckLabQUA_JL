import pickle
import numpy as np
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_amp_cal_1Q import AmpCal1Q
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_RB1Q_simul import RB1QSimul
from LabCode2.Experiments.m_RB1Q import RB1Q
from LabCode2.Experiments.m_amp_cal_2Q import AmpCal2Q

from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ5A2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']

LOs = [Q1['q_LO'], Q1['r_LO'], Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

# Single shot calibration
shots = 1000

SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')

calib_params_1 = SS_1.calib_params
calib_params_2 = SS_2.calib_params

num_ms = 9
num_ks = 1000
num_ns = 1

fidels = np.zeros([2,20])

for i in range(20):

    RB_1 = RB1Q('q1',num_ms,num_ks,num_ns,calib_params_1,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
    RB_1.acquire_RB(interleaved=False)
    RB_1.analyze()

    RB_2 = RB1Q('q2',num_ms,num_ks,num_ns,calib_params_2, path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
    RB_2.acquire_RB(interleaved=False)
    RB_2.analyze()

    RB = RB1QSimul('q1', 'q2', num_ms, num_ks, num_ns, calib_params_1, calib_params_2, path="RB", outer_folder=outer_folder, meta_dict=meta_dict)
    RB.acquire_RB()
    RB.analyze()

    fidels[0][i] = RB_2.gate_fidel
    fidels[1][i] = RB.gate_fidel_2

    print(fidels[0])
    print(fidels[1])

print(fidels)