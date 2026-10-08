import pickle
import math
from LabCode2.Config.make_meta_dict import make_meta_dict
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_amp_cal_1Q import AmpCal1Q
from LabCode2.Experiments.m_power_funnel import PowerFunnel
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_Rabi_1120 import Rabi_1120
from LabCode2.Experiments.m_amp_cal_2Q import AmpCal2Q

from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q

def nearest_kHz(f):
    return int(math.ceil(f/1e3))*1e3

def fine_tune_1QRB(meta_dict, outer_folder):

    # Single shot calibration
    #
    shots = 1000

    SS_1 = SingleShot1Q(q, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix=q)
    # SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')
    # SS_c = SingleShot1Q('qc', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Qc') # work in progress

    calib_params_1 = SS_1.calib_params
    # calib_params_2 = SS_2.calib_params
    # calib_params_c = SS_c.calib_params

    # T2R
    t_min = 100e-9
    t_max = 30e-6#30e-6
    t_max_2 = 0.5e-6
    num_t = 201#50
    shots = 100
    detuning = 0.2e6

    # meta_dict[q]['q_LO']['LO_freq'] += -1e6
    # meta_dict['q2']['q_LO']['LO_freq'] += -1e6
    LOs = [meta_dict[q]['q_LO']]#, meta_dict['q2']['q_LO'], meta_dict['qc']['q_LO']]
    # set_LO_parameters(LOs)

    freq_cal_1 = T2R(q, t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                     path='tuning', suffix=f'{q}_f',freq_cal=True)
    # freq_cal_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
    #                  path='tuning', suffix='Q2_f',freq_cal=True)
    # # freq_cal_c = T2R('qc', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict,
    # #                  outer_folder=outer_folder,
    #                  path='tuning', suffix='Qc_f', freq_cal=True)

    meta_dict[q]['q_LO']['LO_freq'] = nearest_kHz(meta_dict[q]['q_LO']['LO_freq'] - freq_cal_1.freq_correction)
    # meta_dict['q2']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['q2']['q_LO']['LO_freq'] - freq_cal_2.freq_correction)
    # # meta_dict['qc']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['qc']['q_LO']['LO_freq'] - freq_cal_c.freq_correction)

    LOs = [meta_dict[q]['q_LO']]#, meta_dict['q2']['q_LO'], meta_dict['qc']['q_LO']]
    # set_LO_parameters(LOs)

    freq_cal_1 = T2R(q, t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                path='tuning', suffix=f'{q}_f', freq_cal=True)
    # freq_cal_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
    #             path='tuning', suffix='Q2_f', freq_cal=True)
    # freq_cal_c = T2R('qc', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict,
    #                  outer_folder=outer_folder,
    #                  path='tuning', suffix='Qc_f', freq_cal=True)

    meta_dict[q]['q_LO']['LO_freq'] = nearest_kHz(meta_dict[q]['q_LO']['LO_freq'] - freq_cal_1.freq_correction)
    # meta_dict['q2']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['q2']['q_LO']['LO_freq'] - freq_cal_2.freq_correction)
    # meta_dict['qc']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['qc']['q_LO']['LO_freq'] - freq_cal_c.freq_correction)

    LOs = [meta_dict[q]['q_LO']]#, meta_dict['q2']['q_LO'], meta_dict['qc']['q_LO']]
    # set_LO_parameters(LOs)

    # Power funnel
    err_range = 0.35
    num_errs = 61
    num_pis = 41
    shots = 500

    # Power funnel
    funnel_1 = PowerFunnel(q, err_range, num_errs, num_pis, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                         path='tuning', suffix=f'{q}_p')
    # funnel_2 = PowerFunnel('q2', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
    #                        path='tuning', suffix='Q2_p')
    # funnel_c = PowerFunnel('qc', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict,
    #                        outer_folder=outer_folder,
    #                        path='tuning', suffix='Qc_p')

    meta_dict[q]['pi_amp'] = funnel_1.power_factor * meta_dict[q]['pi_amp']
    # meta_dict[q]['pi2_amp'] = meta_dict[q]['pi_amp'] / 2

    # meta_dict['q2']['pi_amp'] = funnel_2.power_factor * meta_dict['q2']['pi_amp']
    # meta_dict['q2']['pi2_amp'] = meta_dict['q2']['pi_amp'] / 2

    # meta_dict['qc']['pi_amp'] = funnel_c.power_factor * meta_dict['qc']['pi_amp']
    # meta_dict['qc']['pi2_amp'] = meta_dict['qc']['pi_amp'] / 2


    # Amp cal
    err_range = 0.05
    num_errs = 31
    num_pis = 10

    amp_cal_1 = AmpCal1Q(q, err_range, num_errs, num_pis, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                         path='tuning', suffix=f'{q}_a')
    # amp_cal_2 = AmpCal1Q('q2', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
    #                      path='tuning', suffix='Q2_a')
    # amp_cal_c = AmpCal1Q('qc', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict,
    #                      outer_folder=outer_folder,
    #                      path='tuning', suffix='Qc_a')


    meta_dict[q]['pi_amp'] = amp_cal_1.power_factor * meta_dict[q]['pi_amp']
    # meta_dict[q]['pi2_amp'] = meta_dict[q]['pi_amp'] / 2

    # meta_dict['q2']['pi_amp'] = amp_cal_2.power_factor * meta_dict['q2']['pi_amp']
    # # meta_dict['q2']['pi2_amp'] = meta_dict['q2']['pi_amp'] / 2

    # meta_dict['qc']['pi_amp'] = amp_cal_c.power_factor * meta_dict['qc']['pi_amp']
    # meta_dict['qc']['pi2_amp'] = meta_dict['qc']['pi_amp'] / 2

    err_range = 0.02
    num_errs = 31
    num_pis = 20

    amp_cal_1 = AmpCal1Q(q, err_range, num_errs, num_pis, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                         path='tuning', suffix=f'{q}_a')
    # amp_cal_2 = AmpCal1Q('q2', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
    #                      path='tuning', suffix='Q2_a')
    # amp_cal_c = AmpCal1Q('qc', err_range, num_errs, num_pis, shots, calib_params_2, meta_dict=meta_dict,
    #                      outer_folder=outer_folder,
    #                      path='tuning', suffix='Qc_a')

    meta_dict[q]['pi_amp'] = amp_cal_1.power_factor * meta_dict[q]['pi_amp']
    # meta_dict[q]['pi2_amp'] = meta_dict[q]['pi_amp'] / 2

    # meta_dict['q2']['pi_amp'] = amp_cal_2.power_factor * meta_dict['q2']['pi_amp']
    # meta_dict['q2']['pi2_amp'] = meta_dict['q2']['pi_amp'] / 2

    # meta_dict['qc']['pi_amp'] = amp_cal_c.power_factor * meta_dict['qc']['pi_amp']
    # meta_dict['qc']['pi2_amp'] = meta_dict['qc']['pi_amp'] / 2

    return meta_dict

def fine_tune_2QRB(meta_dict, outer_folder):

    meta_dict = fine_tune_1QRB(meta_dict, outer_folder)

    q6 = meta_dict[q]
    # Q2 = meta_dict['q2']
    # coupler = meta_dict['c1']

    config = meta_dict_to_config_2Q(meta_dict)

    shots = 1000

    SS_1 = SingleShot1Q(meta_dict[q], shots, cfg=config, outer_folder=outer_folder, path='SingleShotCal',
                        suffix=q)

    calib_params_1 = SS_1.calib_params

    t_min = 16e-9
    t_max = 1e-6
    num_t = 50
    shots = 1000

    CZ = Rabi_1120(q6, Q2, coupler, t_min, t_max, num_t, shots, calib_params_1, cfg=config, suffix='CZ_r', path='tuning', outer_folder=outer_folder)

    pulse_time = CZ.pulse_time
    meta_dict['global_params']['iSWAP_len'] = pulse_time
    config = meta_dict_to_config_2Q(meta_dict)

    # err_range = 0.1
    # num_errs = 21
    # num_pis = 10
    #
    # CZ_amp_cal = AmpCal2Q(q6,Q2,coupler,err_range,num_errs,num_pis,shots,calib_params_1,outer_folder=outer_folder,path='tuning', suffix='CZ_a', cfg=config)


if __name__ == '__main__':
    meta_dict = make_meta_dict()
    outer_folder = meta_dict["outer_folder"]

    # meta_dict['global_params']['read_len'] = 1000
    q = "q4"
    Q = meta_dict[q]
    # Q2 = meta_dict['q2']
    # Qc = meta_dict['qc']

    LOs = [Q['q_LO'], Q['r_LO']]#, Q2['q_LO'], Q2['r_LO'], Qc['q_LO'], Qc['r_LO']]

    # set_LO_parameters(LOs)

    meta_dict = fine_tune_1QRB(meta_dict, outer_folder)

    # meta_dict = fine_tune_2QRB(meta_dict, outer_folder)

    # with open(pathPickle, 'wb') as f:
    #     pickle.dump(meta_dict,f)

    print(meta_dict)
    print(q)
    print(meta_dict[q]['q_LO']['LO_freq'])
    print(meta_dict[q]['pi_amp'])
    # print('q2')
    # print(meta_dict['q2']['q_LO']['LO_freq'])
    # print(meta_dict['q2']['pi_amp'])
    # print('qc')
    # print(meta_dict['qc']['q_LO']['LO_freq'])
    # print(meta_dict['qc']['pi_amp'])