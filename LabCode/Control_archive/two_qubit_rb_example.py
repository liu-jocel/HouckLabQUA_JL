import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from qm.qua import *
from qualang_tools.bakery.bakery import Baking
from LabCode2.RB import TwoQubitRb

import pickle
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.Helpers.qua_helpers import measure_2q

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ5A2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

config = meta_dict_to_config_2Q(meta_dict)

def bake_phased_xz(baker: Baking, q, x, z, a):
    qe = 'q1' if q == 0 else 'q2'
    pulse = 'X180'

    baker.frame_rotation_2pi(-a, qe)
    baker.play(pulse, qe, amp=x)
    baker.frame_rotation_2pi(a+z, qe)


# def bake_sqrt_iswap(baker: Baking, q1, q2):
#     baker.play(iswap_pulse, qubit0_flux_qe)

def bake_cz(baker: Baking,q1,q2):
    baker.play('iSWAP_const', 'c1')
    baker.frame_rotation_2pi(-0.72, 'q1')
    baker.frame_rotation_2pi(-0.09, 'q2')

def prep():
    wait(10000)


def meas():
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    measure('read', 'r1', None, demod.full('integ_W1_' + 'r1', I1, 'out'),
            demod.full('integ_W2_' + 'r1', Q1, 'out'))
    measure('read', 'r2', None, demod.full('integ_W1_' + 'r2', I2, 'out'),
            demod.full('integ_W2_' + 'r2', Q2, 'out'))
    return I1, Q1, I2, Q2

# %%

shots = 1000
SS = SingleShot2Q(meta_dict['q1'], meta_dict['q2'], shots, cfg=config, outer_folder=outer_folder, path='SingleShotCal', suffix='2Q')
calib_params = SS.calib_params

qmm = QuantumMachinesManager(host='192.168.1.127', port=80)

rb = TwoQubitRb(config, bake_phased_xz, {"CZ": bake_cz}, prep, meas)

res = rb.run(qmm, sequence_depths=[1, 5, 10], num_repeats=40, num_averages=100)


def job_to_shots_2qrb(job, num_seqs,num_repeats,num_averages, calib_params):
    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    I_1 = res_handles.get("I_c1").fetch_all()['value']
    Q_1 = res_handles.get("Q_c1").fetch_all()['value']
    I_2 = res_handles.get("I_c2").fetch_all()['value']
    Q_2 = res_handles.get("Q_c2").fetch_all()['value']

    ss_thresh_1 = calib_params[0]
    ss_factor_1 = calib_params[1]
    ss_theta_1 = calib_params[2]
    ss_thresh_2 = calib_params[3]
    ss_factor_2 = calib_params[4]
    ss_theta_2 = calib_params[5]

    IQ_1 = ss_factor_1 * np.real(np.exp(-1j * ss_theta_1) * (I_1 + 1j * Q_1))
    IQ_2 = ss_factor_2 * np.real(np.exp(-1j * ss_theta_2) * (I_2 + 1j * Q_2))
    IQ_10_1 = np.zeros(len(IQ_1))
    IQ_10_2 = np.zeros(len(IQ_2))

    for i, v in enumerate(IQ_1):
        if v > ss_thresh_1:
            IQ_10_1[i] = 1

    for i, v in enumerate(IQ_2):
        if v > ss_thresh_2:
            IQ_10_2[i] = 1

    IQ_10 = np.zeros(len(IQ_1), dtype=int)

    for i in range(len(IQ_1)):
        IQ_10[i] = IQ_10_1[i] + 2 * IQ_10_2[i]

    ss_array = IQ_10.reshape([num_seqs,num_repeats,num_averages])

    return ss_array

counts = job_to_shots_2qrb(res,3,40,100,calib_params)

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()

