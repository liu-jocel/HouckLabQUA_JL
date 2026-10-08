import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# NOTE: this is compatible with Python 3.10, but may not be with older versions of Python

# run this if you run into "module not found" errors - warning, this is a hotfix
import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_6Q
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Coherence.m_T1_RAW import T1_RAW
from LabCode.Experiments.Coherence.m_INTERLEAVED_STARK_RAW import INTERLEAVED_STARK_RAW
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from qualang_tools.units import unit
from qualang_tools.results import fetching_tool
from LabCode.Helpers.qm_result_fetcher import QMResultFetcher
import h5py
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
u = unit(coerce_to_integer=True)

"""
Single qubit coherence measurements that save raw shots to hdf5.
"""
QUBIT = "q1"
main_dict = {QUBIT: {
        'T1_Stark': {
            'run': True,
            'stark_detuning': 50 * u.MHz,
            'amp_max': 0.45, # sweeps negative to positive (negative amp means negative detuning)
            'taus': [100 * u.us],
            'sweep_points': 600
        },
        'T1': {
            'run': True,
            't_min': 1 * u.us,
            't_max': 3000 * u.us,
            'sweep_points': 71
        },
        'T2R': {
            'run': False,
            't_min': 16 * u.ns,
            't_max': 2 * u.us,
            'detuning': 0.5 * u.MHz,
            'sweep_points': 30
        },
        'T2E': {
            'run': False,
            't_min': 1 * u.us,
            't_max': 2000 * u.us,
            'sweep_points': 30
        },
        'T2CPMG': {
            'run': False,
            't_max': [1000 * u.us] * 9,
            'pi_pulses': [3, 7, 21, 61, 241, 501, 1001, 2501, 4001],
            'sweep_points': 60
        }
},
}

FETCH_INTERVAL_SECONDS = 5
TOTAL_EXPERIMENT_TIME_SECONDS = 120 * 60 * 60
SS_CAL_SHOTS = 1000
MINUTES_PER_SS_CAL = 30

meta_dict = make_meta_dict_6Q()
OUTER_FOLDER = meta_dict[QUBIT]['outer_folder']
experiment_start_time = time.time()

def get_amp_max(experiment_dict):
    amp_max = 0
    for experiment, experiment_params in experiment_dict.items():
        if experiment == "T1_Stark" and experiment_params['run']:
            if experiment_params['amp_max'] > amp_max:
                amp_max = experiment_params['amp_max']
    return amp_max

def run_ss_calib(qubit, meta_dict, config, suffix="SS_Cal"):
    fidelity = 0
    SS = None
    while fidelity < 0.75:
        SS = SingleShot1Q(qubit, SS_CAL_SHOTS, meta_dict=meta_dict,
                outer_folder=OUTER_FOLDER, path=qubit,
                suffix=suffix, config=config)
        fidelity = SS.max_F
    return SS.calib_params

def start_exp_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config):
    sweep_vec_dict = {}
    stark_t1_detuning = 50
    ramsey_detuning = 0

    for experiment, experiment_params in experiment_dict.items():
        if experiment == "T1_Stark" and experiment_params['run']:
            stark_t1_detuning = experiment_params['stark_detuning']
            for i, tau in enumerate(experiment_params['taus']): # note that the raw value of tau is in ns
                # note that negative amplitude is used to represent negative detuning in the downstream code
                sweep_vec = np.linspace(-experiment_params['amp_max'], experiment_params['amp_max'], experiment_params['sweep_points'])
                sweep_vec_dict[f'T1Stark_{tau}'] = sweep_vec
        elif experiment == "T1" and experiment_params['run']:
            t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']), experiment_params['sweep_points'])
            sweep_vec_dict['T1'] = t_vec
        elif experiment == "T2R" and experiment_params['run']:
            t_vec = np.linspace(experiment_params['t_min'], experiment_params['t_max'], experiment_params['sweep_points'])
            sweep_vec_dict['T2R'] = t_vec
            ramsey_detuning = experiment_params['detuning']
        elif experiment == "T2E" and experiment_params['run']:
            t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']), experiment_params['sweep_points'])
            sweep_vec_dict['T2E'] = t_vec
        elif experiment == "T2CPMG" and experiment_params['run']:
            for i, pi_pulses in enumerate(experiment_params['pi_pulses']):
                t_min = (16 + pi_pulses * 20) * u.ns
                t_max = experiment_params['t_max'][i]
                t_vec = np.linspace(t_min, t_max, experiment_params['sweep_points'])
                #t_vec = np.logspace(np.log10(t_min), np.log10(t_max), t_points)
                sweep_vec_dict[f'T2CPMG_{pi_pulses}'] = t_vec

    MD = copy.deepcopy(meta_dict[QUBIT][QUBIT])

    amp_max = get_amp_max(experiment_dict)
    MD['cw_amp'] = amp_max
    config_copy = meta_dict_to_config_1Q(MD)
    return INTERLEAVED_STARK_RAW(qubit, sweep_vec_dict, new_calib_params,
                                 stark_t1_detuning=stark_t1_detuning, ramsey_detuning = ramsey_detuning,
                                 amp_max=amp_max, meta_dict=MD, config=config_copy)

def get_exp_list(experiment_dict):
    """
    Construct list of experiments and list of experiments for filename
    These differ since filename does not list the number of pi-pulses for CPMG
    """
    exp_list = []
    filename_exp_list = []
    for exp, exp_params in experiment_dict.items():
        if exp == 'T1_Stark' and exp_params['run']:
            filename_exp_list.append('T1Stark')
            for tau in exp_params['taus']:
                exp_list.append(f'T1Stark_{int(tau / 1000)}us')
        elif exp == 'T2CPMG' and exp_params['run']:
            filename_exp_list.append('T2CPMG')
            for num_pi in exp_params['pi_pulses']:
                exp_list.append(f'T2CPMG_{num_pi}')
        elif exp_params['run']:
            exp_list.append(exp)
            filename_exp_list.append(exp)
    filename = "_".join(filename_exp_list)
    return filename, exp_list

def make_new_h5_file(qubit, experiment_dict, ss_calib_params, job):
    datetime_now = job.time_added.astimezone(ZoneInfo("America/New_York"))
    time_string = datetime_now.strftime('%H_%M_%S_%f')
    date_string = datetime_now.strftime('%Y_%m_%d')

    filename, exp_list = get_exp_list(experiment_dict)
    print(exp_list)
    save_file_path = Path(OUTER_FOLDER) / qubit / f"{qubit}_{date_string}_raw" / f"{time_string}_{filename}_RAW.hdf5"
    save_file_path.parent.mkdir(exist_ok=True)

    num_sweep_pts = 0
    # calculate total number of sweep points over all experiments
    for exp, exp_params in experiment_dict.items():
        if exp == 'T1_Stark' and exp_params['run']:
            num_sweep_pts += exp_params['sweep_points'] * len(exp_params['taus'])
        elif exp == 'T2CPMG' and exp_params['run']:
            num_sweep_pts += exp_params['sweep_points'] * len(exp_params['pi_pulses'])
        elif exp_params['run']:
            num_sweep_pts += exp_params['sweep_points']

    with h5py.File(save_file_path, "a") as file:
        file.create_dataset("I", shape=(0, num_sweep_pts), maxshape=(None, num_sweep_pts), dtype=np.float16)
        file.create_dataset("Q", shape=(0, num_sweep_pts), maxshape=(None, num_sweep_pts), dtype=np.float16)
        file.create_dataset("ss", shape=(0, num_sweep_pts), maxshape=(None, num_sweep_pts), dtype=np.uint8)
        file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)
        file.create_dataset("amp_pts", shape=(num_sweep_pts,), dtype=np.float64)
        file.create_dataset("tau_pts", shape=(num_sweep_pts,), dtype=np.uint64)
        file.create_dataset("exp_order", shape=(num_sweep_pts,), dtype=np.int8)
        file.create_dataset("param_order", shape=(num_sweep_pts,), dtype=np.uint32)

        file.attrs["start"] = f"{date_string}_{time_string}"
        file.attrs["exp_list"] = exp_list
        file.attrs["amp_max"] = get_amp_max(experiment_dict)

        for key, value in ss_calib_params.items():
            file.attrs[key] = value

    return save_file_path


# Notes: need to store the experiment sequence in the file somewhere
for qubit, experiment_dict in main_dict.items():

    qubit_meta_dict = meta_dict[qubit][qubit]

    # Set LO

    #SS Calibration
    config = meta_dict_to_config_1Q(qubit_meta_dict)
    new_calib_params = run_ss_calib(qubit, qubit_meta_dict, config)
    qubit_meta_dict['read_theta'] = new_calib_params["read_theta"]
    config = meta_dict_to_config_1Q(qubit_meta_dict)
    time_since_last_ss_calib = time.time()

    # start main experiment loop

    exp_raw = start_exp_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config)
    job = exp_raw.job
    fetcher = QMResultFetcher(job.result_handles)

    save_file_path = make_new_h5_file(qubit, experiment_dict, new_calib_params, job)

    total_elapsed_time = time.time() - experiment_start_time
    new_data = {}
    time.sleep(FETCH_INTERVAL_SECONDS)
    while not fetcher.is_done_fetching:

        # check if it is time to stop running this experiment
        if total_elapsed_time > TOTAL_EXPERIMENT_TIME_SECONDS:
            job.halt()

        # fetch new data if available
        new_data = fetcher.fetch()
        last_count, new_count = fetcher.counts
        print(last_count, new_count)
        report = job.execution_report()
        print(report)

        if new_data:
            # write new data to h5, resize, flush
            with h5py.File(save_file_path, "a") as file:
                dataset_keys = {"I", "Q", "ss"}

                for key in dataset_keys:
                    dataset = file[key]
                    dataset.resize(new_count, axis=0)
                    dataset[last_count:new_count] = new_data[key]
                    dataset.flush()

                amp_pts_dataset = file["amp_pts"]
                amp_pts = new_data["amp_pts"]
                amp_pts_dataset[:] = amp_pts
                amp_pts_dataset.flush()

                tau_pts_dataset = file["tau_pts"]
                tau_pts = new_data["tau_pts"]
                tau_pts_dataset[:] = tau_pts
                tau_pts_dataset.flush()

                exp_order_dataset = file["exp_order"]
                exp_order = new_data["exp_order"]
                exp_order_dataset[:] = exp_order
                exp_order_dataset.flush()

                param_order_dataset = file["param_order"]
                param_order = new_data["param_order"]
                param_order_dataset[:] = param_order
                param_order_dataset.flush()

                # only save the first timestamp for each repetition N of the T1 sequence
                ts_dataset = file["timestamps"]
                timestamps = new_data["timestamps"].T[0]
                ts_dataset.resize(new_count, axis=0)
                # print(last_count, new_count, taus)
                ts_dataset[last_count:new_count] = timestamps
                ts_dataset.flush()

        # check if ss time has passed, if so,
        # then halt job, do ss, restart t1 tracking in new h5 file
        time_elapsed_since_last_ss_calib = time.time() - time_since_last_ss_calib
        if time_elapsed_since_last_ss_calib > MINUTES_PER_SS_CAL * 60:
            job.halt()
            new_calib_params = run_ss_calib(qubit, qubit_meta_dict, config)
            qubit_meta_dict['read_theta'] = new_calib_params["read_theta"]
            config = meta_dict_to_config_1Q(qubit_meta_dict)

            # restart job and point to new hdf5 file
            exp_raw = start_exp_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config)
            job = exp_raw.job
            fetcher = QMResultFetcher(job.result_handles)

            save_file_path = make_new_h5_file(qubit, experiment_dict, new_calib_params, job)
            new_data = {}

            time_since_last_ss_calib = time.time()

        time.sleep(FETCH_INTERVAL_SECONDS)
        total_elapsed_time = time.time() - experiment_start_time
