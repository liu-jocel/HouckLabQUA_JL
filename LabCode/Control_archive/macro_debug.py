import pickle
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.Helpers.gate_macros import CZ, flat_top_CZ
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ5A2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
coupler = meta_dict['c1']
meta_dict['c1']['c_IF'] = 0
config = meta_dict_to_config_2Q(meta_dict)

meas_pulses = [('Zero', 'Zero'),
                ('Zero', 'Ym90'),
                ('Zero', 'X90'),
                ('Ym90', 'Zero'),
                ('Ym90', 'Ym90'),
                ('Ym90', 'X90'),
                ('X90', 'Zero'),
                ('X90', 'Ym90'),
                ('X90', 'X90')]

with program() as prog:
    n = declare(int)

    flat_top_CZ(coupler,75)

qmm = QuantumMachinesManager(host='192.168.1.127', port='80')


def simulate(prog, channels, ticks, ns_to_plot):
    qm = qmm.open_qm(config)
    qmm.clear_all_job_results()

    job = qm.simulate(prog, SimulationConfig(duration=ticks))
    samples = job.get_simulated_samples()

    # get channel traces
    traces = []
    for c in channels:
        traces.append(samples.con1.analog[c])

    assert len(ns_to_plot) <= 4, 'Too many time windows to plot.'

    # make axis object
    if len(ns_to_plot) == 1:
        ax_shape = [1, 1]
        fig, ax = plt.subplots(ax_shape[0], ax_shape[1], constrained_layout=True)
        ax = [[ax]]
    elif len(ns_to_plot) == 2:
        ax_shape = [1, 2]
        fig, ax = plt.subplots(ax_shape[0], ax_shape[1], constrained_layout=True)
        ax = [ax]
    else:
        ax_shape = [2, 2]
        fig, ax = plt.subplots(ax_shape[0], ax_shape[1], constrained_layout=True)
    # make plots
    for i in range(ax_shape[0]):
        for j in range(ax_shape[1]):
            start_ns = ns_to_plot[i * ax_shape[1] + j][0]
            stop_ns = ns_to_plot[i * ax_shape[1] + j][1]
            ns_axis = np.linspace(start_ns, stop_ns, stop_ns - start_ns)
            for k in range(len(channels)):
                trace_to_plot = traces[k][start_ns:stop_ns]
                ax[i][j].plot(ns_axis, trace_to_plot, label=channels[k])
            ax[i][j].legend()
            ax[i][j].set_ylabel('Output (V)')
            ax[i][j].set_xlabel('Experiment time (ns)')

    plt.savefig('CZ/test_sim', bbox_inches="tight")
    plt.close(fig)

    return samples

simulate(prog,['1','3','9'],1000,[[100,1000],[200,250]])