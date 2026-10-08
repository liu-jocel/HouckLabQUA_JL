import pickle
import numpy as np
import matplotlib
import atexit
import matplotlib.pyplot as plt
import tkinter as tk

def cleanup():
    try:
        plt.close("all")
        if tk._default_root:
            tk._default_root.quit()
            tk._default_root.destroy()
    except Exception:
        pass

atexit.register(cleanup)
import matplotlib.pyplot as plt
import time
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_6Q
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Coherence.m_T1_RAW import T1_RAW
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

main_dict = {'q2': {
        'T1_RAW': {
            'run': True,
            't_min': 1 * u.us,
            't_max': 1500 * u.us,
            't_points': 2,
            'object': T1_RAW,
    },
},
}

FETCH_INTERVAL_SECONDS = 5
TOTAL_EXPERIMENT_TIME_SECONDS = 2 * 60 * 60
SS_CAL_SHOTS = 1000
MINUTES_PER_SS_CAL = 30

meta_dict = make_meta_dict_6Q()
OUTER_FOLDER = meta_dict['q2']['outer_folder']
experiment_start_time = time.time()

def run_ss_calib(qubit, meta_dict, config, suffix="SS_Cal"):
    fidelity = 0
    SS = None
    while fidelity < 0.8:
        SS = SingleShot1Q(qubit, SS_CAL_SHOTS, meta_dict=meta_dict,
                outer_folder=OUTER_FOLDER, path=qubit,
                suffix=suffix, config=config)
        fidelity = SS.max_F
    return SS.calib_params

def start_t1_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config):
    for experiment, experiment_params in experiment_dict.items():
        if experiment == "T1_RAW" and experiment_params['run']:
            t_vec = [20*u.ns, 500*u.us]#np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                # experiment_params['t_points'])
            return T1_RAW(qubit, t_vec, new_calib_params, meta_dict=qubit_meta_dict, config=config)

def make_new_h5_file(qubit, experiment_dict, ss_calib_params, job):
    datetime_now = job.time_added.astimezone(ZoneInfo("America/New_York"))
    time_string = datetime_now.strftime('%H_%M_%S_%f')
    date_string = datetime_now.strftime('%Y_%m_%d')
    save_file_path = Path(OUTER_FOLDER) / qubit / f"{qubit}_{date_string}" / f"{time_string}_T1_RAW.hdf5"
    save_file_path.parent.mkdir(exist_ok=True)

    num_taus = experiment_dict["T1_RAW"]["t_points"]
    with h5py.File(save_file_path, "a") as file:
        file.create_dataset("I", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.float16)
        file.create_dataset("Q", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.float16)
        file.create_dataset("ss", shape=(0, num_taus), maxshape=(None, num_taus), dtype=np.uint8)
        file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)
        file.create_dataset("tau", shape=(num_taus,), dtype=np.uint64)

        file.attrs["start"] = f"{date_string}_{time_string}"

        for key, value in ss_calib_params.items():
            file.attrs[key] = value

    return save_file_path

for qubit, experiment_dict in main_dict.items():

    qubit_meta_dict = meta_dict[qubit][qubit]

    # Set LO
    LOs = [qubit_meta_dict['q_LO'], qubit_meta_dict['r_LO']]
    print(f"{qubit} LO: {LOs}")
    # set_LO_parameters(LOs)

    #SS Calibration
    config = meta_dict_to_config_1Q(qubit_meta_dict)
    new_calib_params = run_ss_calib(qubit, qubit_meta_dict, config)
    qubit_meta_dict['read_theta'] = new_calib_params["read_theta"]
    config = meta_dict_to_config_1Q(qubit_meta_dict)
    time_since_last_ss_calib = time.time()

    # start main experiment loop

    t1_raw = start_t1_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config)
    job = t1_raw.job
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
            t1_raw = start_t1_raw(qubit, experiment_dict, new_calib_params, qubit_meta_dict, config)
            job = t1_raw.job
            fetcher = QMResultFetcher(job.result_handles)

            save_file_path = make_new_h5_file(qubit, experiment_dict, new_calib_params, job)
            new_data = {}

            time_since_last_ss_calib = time.time()

        time.sleep(FETCH_INTERVAL_SECONDS)
        total_elapsed_time = time.time() - experiment_start_time
