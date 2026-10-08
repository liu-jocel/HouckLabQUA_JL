from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from pathlib import Path
import matplotlib
from qualang_tools.bakery import baking
matplotlib.use("QtAgg") # if you want interactive plots
import matplotlib.pyplot as plt
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Helpers.pulse_macros import *
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
from LabCode.Helpers.pulse_macros import drag_cosine_pulse_waveforms


qubit ='q1'
meta_dict = make_meta_dict(qubit)[qubit]
IF_freq = meta_dict['IF_freq']
anharmonicity = meta_dict['anharmonicity']
resonator = meta_dict['r_name']
config = meta_dict_to_config_1Q(meta_dict)
qubit_freq = -200*u.MHz
stark_detuned_freq = -150*u.MHz
# Connect to the local Quantum Machines Manager
qmm = QuantumMachinesManager(host="192.168.0.134")    #OXFORD
# qmm = QuantumMachinesManager(host="192.168.0.188")  # EBAY
pulse_type = "X90"

print(IF_freq + anharmonicity)
# Define your QUA program
with program() as qua_simulation:
    # Declare a variable for time tagging or measurement if needed
    # frame_rotation_2pi(0.5, qubit)
    # update_correction(qubit, 1, 0, 0, 1)
    play('X90', qubit)
    frame_rotation_2pi(0.5, qubit)
    play('Y90', qubit)


# -------------------------------------------------------------
# Simulation settings
# -------------------------------------------------------------
simulation_duration = 300  # in clock cycles (1 clock cycle = 4 ns)
simulation_config = SimulationConfig(duration=simulation_duration)

# Open the quantum machine
qm = qmm.open_qm(config)

# Run the simulation
job = qm.simulate(qua_simulation, simulation_config)

# -------------------------------------------------------------
# Retrieve simulated samples
# -------------------------------------------------------------
samples = job.get_simulated_samples()
# samples.con1.plot()  # plots controller 1 analog output
waveform_report = job.get_simulated_waveform_report()
waveform_dict = waveform_report.to_dict()
print(waveform_dict.keys())
waveform_report.create_plot(samples, plot=True,)

