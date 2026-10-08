from config import gen_config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from PythonDrivers.control_8257 import set_LO_parameters
from Config.config_helpers import meta_dict_to_config_2Q
import numpy as np
import pickle
# from LabCode.Calib.initialize import *
import datetime

datetimenow = datetime.datetime.now()
datestring = datetimenow.strftime("%Y_%m_%d")
pathPickle = "Y:/Charlie/Data/TwoQubit/TQ5A2/meta_dict/meta_dict.pkl"

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# Define element to test
qubit = meta_dict['c1']
element = 'c1'
LOs = [meta_dict['q1']['q_LO'], meta_dict['q2']['q_LO'], meta_dict['c1']['c_LO']]

# set_LO_parameters(LOs)

config = meta_dict_to_config_2Q(meta_dict)

with program() as pulse_test:

    n = declare(int)
    with for_(n, 0, n < 10, n + 1):
    # with infinite_loop_():
        wait_for_trigger('q1')
        align()
        # reset_phase('q1')
        # reset_phase('q2')
        # reset_phase('c1')
        reset_frame('q1','q2','c1')
        frame_rotation_2pi(0.5,'c1')
        play('X180', 'q2')
        play('X180', 'q1')
        play('iSWAP_const', 'c1')

        # wait(int(250e6))


with program() as stop_playing:
    wait(16)

# Can test by making the pulse the zero waveform and check that it goes away
qmm = QuantumMachinesManager(host='192.168.1.127', port='80')
qm = qmm.open_qm(config)
qmm.clear_all_job_results()

# job = qm.execute(stop_playing)
job = qm.execute(pulse_test)