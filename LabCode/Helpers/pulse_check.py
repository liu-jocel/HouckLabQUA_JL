from qm.qua import *
from qm import QuantumMachinesManager
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)
import matplotlib.pyplot as plt
from matplotlib import use

use("TkAgg")

qubit = "q1"
# meta_dict = make_meta_dict_CR()
meta_dict = make_meta_dict(qubit)[qubit]
config = meta_dict_to_config_1Q(meta_dict)
Q = meta_dict

LOs = [Q["q_LO"], Q["r_LO"]]

# set_LO_parameters(LOs)

with program() as prog:
    m = declare(int)
    with infinite_loop_():
        # play("X90", qubit)
        play("constant_tone",qubit)
        #wait(10000*u.ns)


# Can test by making the pulse the zero waveform and check that it goes away
qmm = QuantumMachinesManager(host="192.168.0.134") # OXFORD
# qmm = QuantumMachinesManager(host="192.168.0.188")  # EBAY
qm = qmm.open_qm(config)

qmm.clear_all_job_results()

job = qm.execute(prog)
#qm.close()
