import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# NOTE: this is compatible with Python 3.10, but may not be with older versions of Python

# run this if you run into "module not found" errors - warning, this is a hotfix
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_6Q
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Coherence.m_T1_RAW import T1_RAW
from LabCode.Experiments.Coherence.m_INTERLEAVED_RAW import INTERLEAVED_RAW
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
QUBIT = "q6"
t_points = 71
main_dict = {QUBIT: {
        'T1': {
            'run': True,
            't_min': 1 * u.us,
            't_max': 10000 * u.us,
        },
        'T2R': {
            'run': False,
            't_min': 16 * u.ns,
            't_max': 2 * u.us,
            'detuning': 0.5 * u.MHz,
        },
        'T2E': {
            'run': False,
            't_min': 1 * u.us,
            't_max': 3000 * u.us,
        },
        'T2CPMG': {
            'run': False,
            't_max': [2000 * u.us] * 9,
            'pi_pulses': [3, 7, 21, 61, 241, 501, 1001, 2501, 4001],
        }
},
}

FETCH_INTERVAL_SECONDS = 5
TOTAL_EXPERIMENT_TIME_SECONDS = 12 * 60 * 60
SS_CAL_SHOTS = 1000
MINUTES_PER_SS_CAL = 30

meta_dict = make_meta_dict_6Q()
OUTER_FOLDER = meta_dict[QUBIT]['outer_folder']
experiment_start_time = time.time()

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
    t_vec_dict = {}
    detuning = 0
    for experiment, experiment_params in experiment_dict.items():
        if experiment == "T1" and experiment_params['run']:
            t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']), t_points) # [24*u.ns, 500*u.us]#
            t_vec_dict['T1'] = t_vec
        elif experiment == "T2R" and experiment_params['run']:
            t_vec = np.linspace(experiment_params['t_min'], experiment_params['t_max'], t_points) # np.array([24*u.ns, 5.4*u.us])
            t_vec_dict['T2R'] = t_vec
            detuning = experiment_params['detuning']
        elif experiment == "T2E" and experiment_params['run']:
            t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']), t_points) #[24*u.ns, 800*u.us]
            t_vec_dict['T2E'] = t_vec
        elif experiment == "T2CPMG" and experiment_params['run']:
            for i, pi_pulses in enumerate(experiment_params['pi_pulses']):
                t_min = (16 + pi_pulses * 20) * u.ns
                t_max = experiment_params['t_max'][i]
                t_vec = np.linspace(t_min, t_max, t_points)
                #t_vec = np.logspace(np.log10(t_min), np.log10(t_max), t_points)
                t_vec_dict[f'T2CPMG_{pi_pulses}'] = t_vec

    return INTERLEAVED_RAW(qubit, t_vec_dict, new_calib_params, detuning=detuning, meta_dict=qubit_meta_dict, config=config)

def get_exp_list(experiment_dict):
    exp_list = []
    filename_exp_list = []
    for exp, exp_params in experiment_dict.items():
        if exp == 'T2CPMG' and exp_params['run']:
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

    num_taus = t_points * len(exp_list)
    with h5py.File(save_file_path, "a") as file:
        file.create_dataset("I", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.float16)
        file.create_dataset("Q", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.float16)
        file.create_dataset("ss", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.uint8)
        file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)
        file.create_dataset("tau", shape=(num_taus,), dtype=np.uint64)

        file.attrs["start"] = f"{date_string}_{time_string}"
        file.attrs["exp_list"] = exp_list

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
        if new_data:
            # write new data to h5, resize, flush
            with h5py.File(save_file_path, "a") as file:
                dataset_keys = {"I", "Q", "ss"}

                for key in dataset_keys:
                    dataset = file[key]
                    dataset.resize(new_count, axis=0)
                    dataset[last_count:new_count] = new_data[key]
                    dataset.flush()

                tau_dataset = file["tau"]
                taus = new_data["tau"]
                tau_dataset[:] = taus
                tau_dataset.flush()

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
