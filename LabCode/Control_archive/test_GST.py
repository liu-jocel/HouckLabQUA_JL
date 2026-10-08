import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_6Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
# from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_ZZ import ZZ
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Experiments.m_transmission import Transmission
from LabCode2.Experiments.m_qubit_spec import QubitSpec
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
# from LabCode2.Experiments.m_GST_1Q import GST_1Q
from LabCode2.Experiments.m_RB1Q import RB1Q
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Z://t1Team/Data/2024-08-09_cooldown/TATQ01-si_01_qm/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
element = Q2
config = meta_dict_to_config_6Q(meta_dict)

LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO']]
t_min_t2e = 1e-6
t_max_t2e = 500e-6
num_t_t2e = 25
shots_t2e = 1

T2E = T2E('q1', t_min_t2e, t_max_t2e, num_t_t2e, shots_t2e, None, meta_dict=meta_dict, outer_folder=outer_folder, path='GST', suffix='Q1',)

T2E.simulate(T2E.make_prog,['1','2','3', '4'],101000,[[400,401000]])

# rb = RB1Q('q1', 10, 1, 1, None,outer_folder=outer_folder, path='GST', meta_dict=meta_dict)
#
# rb.simulate(rb.make_prog,['1','2','5'],100000,[[100000,103000]])