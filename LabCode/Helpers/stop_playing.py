from qm.qua import *
from qm import QuantumMachinesManager
from LabCode.Config.config_helpers import meta_dict_to_config_6Q
import pickle
import datetime

# Generate config file from pickle
datetimenow = datetime.datetime.now()
datestring = datetimenow.strftime('%Y_%m_%d')
outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# pathPickle = "Z:/Charlie/Data/TwoQubit/TQ4A2/QM/config_RabiDebug.pkl"
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

config = meta_dict_to_config_1Q(meta_dict)

qmm = QuantumMachinesManager(host="192.168.0.134")    #OXFORD
# qmm = QuantumMachinesManager(host="192.168.0.188")  # EBAY
qm = qmm.open_qm(config)
qmm.clear_all_job_results()

# Stop playing program
with program() as stop_playing:
    align('q1', 'r1')
    pass

my_job = qm.execute(stop_playing)