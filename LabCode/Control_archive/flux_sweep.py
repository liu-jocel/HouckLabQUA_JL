import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_ZZ import ZZ
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Control.fine_tuning import fine_tune_1QRB
from LabCode2.Experiments.m_RB1Q import RB1Q
from LabCode2.Experiments.m_RB1Q_simul import RB1QSimul
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ6B2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

LO_adj_1 = 0e6
LO_adj_2 = 0e6
meta_dict['q1']['q_LO']['LO_freq'] += LO_adj_1
meta_dict['q2']['q_LO']['LO_freq'] += LO_adj_2

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
element = Q2
config = meta_dict_to_config_2Q(meta_dict)


LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

shots = 1000

SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')

calib_params_1 = SS_1.calib_params
calib_params_2 = SS_2.calib_params

path = 'flux_sweep'

yoko = 1.135

YOKOGS200.main(yoko)

# tune for new yoko voltage
meta_dict = fine_tune_1QRB(meta_dict, outer_folder)

with open(pathPickle, 'wb') as f:
    pickle.dump(meta_dict, f)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
config = meta_dict_to_config_2Q(meta_dict)

# T1
t_min = 1e-6
t_max = 200e-6
num_t = 30
shots = 1000

T1_1 = T1('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path=path,
          suffix='Q1', fit=False)
T1_2 = T1('q2', t_min, t_max, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path=path,
          suffix='Q2', fit=False)

# ZZ
t_min = 100e-9
t_max = 0.5e-6
t_max_2 = 0.5e-6
num_t = 100
shots = 1000
detuning = 4e6

ZZ_1 = ZZ('q1', 'q2', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
          path=path, suffix='Q1', fit=False)
ZZ_2 = ZZ('q2', 'q1', t_min, t_max, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
          path=path, suffix='Q2', fit=False)

# T2R
t_min = 100e-9
t_max = 40e-6
t_max_2 = 40e-6
num_t = 40
shots = 1000
detuning = 0.1e6

T2R_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
            path=path, suffix='Q1', freq_cal=False, fit=False)
T2R_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
            path=path, suffix='Q2', freq_cal=False, fit=False)

# T2E
t_min = 1e-6
t_max = 200e-6
t_max_2 = 200e-6
num_t = 40
shots = 1000

T2E_1 = T2E('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path=path,
            suffix='Q1', fit=False)
T2E_2 = T2E('q2', t_min, t_max_2, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path=path,
            suffix='Q2', fit=False)

# num_ms = 9
# num_ks = 10000
# num_ns = 1

# RB_1 = RB1Q('q1', num_ms, num_ks, num_ns, calib_params_1, path=path, outer_folder=outer_folder, meta_dict=meta_dict, fit=False)
# RB_1.acquire(interleaved=False)
# RB_1.analyze()
#
# RB_2 = RB1Q(Q2, num_ms, num_ks, num_ns, calib_params_2, path=path, outer_folder=outer_folder, meta_dict=meta_dict, fit=False)
# RB_2.acquire(interleaved=False)
# RB_2.analyze()
#
# RB = RB1QSimul(Q1, Q2, num_ms, num_ks, num_ns, calib_params_1, calib_params_2, path=path, outer_folder=outer_folder,
#                meta_dict=meta_dict, fit=False)
# RB.acquire()
# RB.analyze()

data = {'yoko': yoko,
        'Q1':{'freq': Q1['q_LO']['LO_freq']+Q1['q_IF'],
              'T1': T1_1.data,
              'T2R': T2R_1.data,
              'T2E': T2E_1.data,
              'ZZ': ZZ_1.data},
              # 'RB': RB_1.data,
              # 'RB_simul': RB.data},
        'Q2':{'freq': Q2['q_LO']['LO_freq']+Q2['q_IF'],
              'T1': T1_2.data,
              'T2R': T2R_2.data,
              'T2E': T2E_2.data,
              'ZZ': ZZ_2.data}
              # 'RB': RB_2.data,
              # 'RB_simul': RB.data}
        }

with open(outer_folder+'flux_sweep/data_dummy.pkl','rb') as f:
    sweep_data=pickle.load(f)
sweep_data[yoko] = data
with open(outer_folder+'flux_sweep/data_dummy.pkl','wb') as f:
    pickle.dump(sweep_data,f)
# print(sweep_data)