import os
import json
import numpy as np
import h5py
import pickle
import datetime
from pathlib import Path
import scipy.optimize as sp
import matplotlib.pyplot as plt
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from qm.simulate.credentials import create_credentials
from LabCode.Config.config_helpers import meta_dict_to_config_6Q
from LabCode.Helpers.ss_helpers import find_state
import LabCode.Helpers.fit_functions as fitters
from qualang_tools.results import progress_counter, fetching_tool
import plotly.io as pio

pio.renderers.default = "browser"
# HOST = "192.168.0.188"  #<--EBAY  
HOST = "192.168.0.134"  # <--OXFORD
PORT = "80"
qmm = QuantumMachinesManager(host=HOST,)#QuantumMachinesManager(host="192.168.1.109") #host="192.168.1.109:80")#, port='80'

class MakeFile(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        # self.attrs["_script"] = open(sys.argv[0], 'r').read()
        # if self.mode is not 'r':
        # self.attrs["_script"] = get_script()
        # if not read-only or existing then save the script into the .h5
        # Maybe should take this automatic feature out and just do it when you want to
        # Automatic feature taken out. Caused more trouble than convenience. Ge Yang
        # if 'save_script' in kwargs:
        # save_script = kwargs['save_script']
        # else:
        # save_script = True
        # if (self.mode is not 'r') and ("_script" not in self.attrs) and (save_script):
        # self.save_script()
        self.flush()

    def add(self, key, data):
        data = np.array(data)
        try:
            self.create_dataset(key, shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=str(data.astype(np.float64).dtype))
        except RuntimeError:
            del self[key]
            self.create_dataset(key, shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=str(data.astype(np.float64).dtype))
        self[key][...] = data

class NpEncoder(json.JSONEncoder):
    """ Ensure json dump can handle np arrays """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ExperimentClass:
    """Base class for all experiments"""

    def __init__(self, path='', outer_folder='',
                 suffix='data', meta_dict = None, config = None, **kwargs):
        """ Initializes experiment class
            @param path - directory where data will be stored
            @param suffix - suffix to use when creating data files
            @param config_file - parameters for config file specified are loaded into the class dict
                                 (name relative to expt_directory if no leading /)
                                 Default = None looks for path/prefix.json
            @param **kwargs - by default kwargs are updated to class dict
            also loads InstrumentManager, LivePlotter, and other helpers
        """

        self.__dict__.update(kwargs)
        self.config = config
        self.path = path
        self.outerFolder = outer_folder
        datetime_now = datetime.datetime.now()
        time_string = datetime_now.strftime('%H_%M_%S')
        date_string = datetime_now.strftime('%Y_%m_%d')
        self.date_string = date_string
        self.time_string = time_string
        self.suffix = str(suffix)
        self.meta_dict = meta_dict

        ##### make sure the data directory exists (idempotent + recursive; avoids
        ##### stale is_dir() caching over network shares that can skip creation)
        flat_path = kwargs.get('flat_path', False)

        if flat_path:
            data_dir = os.path.join(self.outerFolder, self.path)
        else:
            data_dir = os.path.join(self.outerFolder, self.path, self.path + "_" + date_string)
        os.makedirs(data_dir, exist_ok=True)

        base = os.path.join(data_dir, self.path + "_" + time_string + "_" + self.suffix)
        self.pname = base + '.pkl'
        self.fname = base + '.h5'
        self.iname = base + '.jpg'
        self.cname = base + '.json'

    # We want this to be meta_dict instead of config file
    def save_config(self):
        if self.cname[:-3] != '.h5':
            with open(self.cname, 'w') as fid:
                json.dump(self.cfg, fid, cls=NpEncoder),
            self.datafile().attrs['config'] = json.dumps(self.cfg, cls=NpEncoder)

    def acquire(self, prog, config=None, return_qm=False):

        if config == None:
            config = meta_dict_to_config_6Q(self.meta_dict)

        ### Open OPX
        # qmm = QuantumMachinesManager(host="192.168.1.111", port="80")
        qm = qmm.open_qm(config)
        qmm.clear_all_job_results()

        ### Run QUA program
        job = qm.execute(prog, duration_limit=0, data_limit=0)

        print(job.execution_report())
        if return_qm:
            return job, qm
        return job
    def simulate(self, prog, config, duration=10_000, show_report=False):
        """
        show_report=True calls figure.show(renderer="browser"), which can hang for minutes
        and produce nothing on machines where webbrowser.open() can't reach a real browser
        (e.g. this sandbox). Report is always saved to an html file instead.
        """
        import time
        simulation_config = SimulationConfig(duration=duration)
        print(f"[simulate] opening QM...", flush=True)
        t0 = time.time()
        qm = qmm.open_qm(config)
        print(f"[simulate] QM opened ({time.time() - t0:.1f}s). Running qm.simulate (duration={duration} cycles)...", flush=True)
        t0 = time.time()
        job = qm.simulate(prog, simulation_config)
        print(f"[simulate] qm.simulate returned ({time.time() - t0:.1f}s). Fetching simulated samples...", flush=True)
        t0 = time.time()
        samples = job.get_simulated_samples()
        print(f"[simulate] samples fetched ({time.time() - t0:.1f}s). Fetching waveform report "
              f"(this call is server-side and can be slow for large/branchy programs)...", flush=True)
        t0 = time.time()
        waveform_report = job.get_simulated_waveform_report()
        print(f"[simulate] waveform report fetched ({time.time() - t0:.1f}s). Building plot...", flush=True)
        t0 = time.time()
        waveform_dict = waveform_report.to_dict()
        report_dir = os.path.dirname(self.iname)
        # create_plot's save_path only controls the directory: it always names the file
        # "waveform_report_{controller}_{job.id}.html" internally, ignoring any basename given.
        waveform_report.create_plot(samples, plot=show_report, save_path=os.path.join(report_dir, "_"))
        saved_reports = [f for f in os.listdir(report_dir) if f.startswith(f"waveform_report_") and job.id in f]
        report_paths = [os.path.join(report_dir, f) for f in saved_reports]
        print(f"[simulate] waveform report saved to {report_paths} ({time.time() - t0:.1f}s).", flush=True)
        return waveform_dict, samples, job



    def simulate_dep(self, prog, channels, ticks, ns_to_plot, config=None):
        # qmm = QuantumMachinesManager(host='houck-3c064b4a.quantum-machines.co', port=443, credentials=create_credentials())
        qm = qmm.open_qm(config)
        qmm.clear_all_job_results()

        job = qm.simulate(prog(), SimulationConfig(duration=ticks))
        samples = job.get_simulated_samples()

        # get channel traces
        traces = []
        for c in channels:
            traces.append(samples.con1.analog[c])

        assert len(ns_to_plot) <= 4, 'Too many time windows to plot.'

        # make axis object
        if len(ns_to_plot) == 1:
            ax_shape = [1, 1]
            fig, ax = plt.subplots(ax_shape[0],ax_shape[1], constrained_layout=True)
            ax = [[ax]]
        elif len(ns_to_plot) == 2:
            ax_shape = [1, 2]
            fig, ax = plt.subplots(ax_shape[0],ax_shape[1], constrained_layout=True)
            ax = [ax]
        else:
            ax_shape = [2, 2]
            fig, ax = plt.subplots(ax_shape[0],ax_shape[1], constrained_layout=True)
        # make plots
        for i in range(ax_shape[0]):
            for j in range(ax_shape[1]):
                start_ns = ns_to_plot[i*ax_shape[1]+j][0]
                stop_ns = ns_to_plot[i*ax_shape[1]+j][1]
                ns_axis = np.linspace(start_ns,stop_ns,stop_ns-start_ns)
                for k in range(len(channels)):
                    trace_to_plot = traces[k][start_ns:stop_ns]
                    ax[i][j].plot(ns_axis, trace_to_plot, label=channels[k])
                ax[i][j].legend()
                ax[i][j].set_ylabel('Output (V)')
                ax[i][j].set_xlabel('Experiment time (ns)')

        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

        return samples


    def get_IQ(self, job, I_name='I', Q_name='Q'):

        I_handle = job.result_handles.get(I_name)
        Q_handle = job.result_handles.get(Q_name)

        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()

        I = I_handle.fetch_all()['value']
        Q = Q_handle.fetch_all()['value']

        return I,Q

    def job_to_shots_1q(self, job, shape, calib_params, I_name='I', Q_name='Q'):

        I,Q = self.get_IQ(job, I_name, Q_name)

        ss_thresh = calib_params[0]
        ss_factor = calib_params[1]
        ss_theta = calib_params[2]

        IQ = ss_factor * np.real(np.exp(-1j * ss_theta) * (I + 1j * Q))
        IQ_10 = np.zeros(len(IQ))

        for i, v in enumerate(IQ):
            if v > ss_thresh:
                IQ_10[i] = 1

        ss_array = IQ_10.reshape(shape)

        return ss_array

    def job_to_shots_2q_IQ(self, job, shape, calib_params):

        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        I_1 = res_handles.get("I1").fetch_all()['value']
        Q_1 = res_handles.get("Q1").fetch_all()['value']
        I_2 = res_handles.get("I2").fetch_all()['value']
        Q_2 = res_handles.get("Q2").fetch_all()['value']

        # ss_thresh_1 = calib_params[0][0]
        # ss_factor_1 = calib_params[0][1]
        # ss_theta_1 = calib_params[0][2]
        # ss_thresh_2 = calib_params[1][0]
        # ss_factor_2 = calib_params[1][1]
        # ss_theta_2 = calib_params[1][2]

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
            if v > ss_thresh_2:  # maybe this sign is wrong
                IQ_10_2[i] = 1

        ss_array_1 = IQ_10_1.reshape(shape)
        ss_array_2 = IQ_10_2.reshape(shape)

        ss_array = [ss_array_1, ss_array_2]

        return ss_array

    def job_to_shots_2q_IQ_bg(self, job, shape, calib_params):

        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        I_1 = res_handles.get("I1").fetch_all()['value']
        Q_1 = res_handles.get("Q1").fetch_all()['value']
        I_2 = res_handles.get("I2").fetch_all()['value']
        Q_2 = res_handles.get("Q2").fetch_all()['value']

        I_1_bg = res_handles.get("I1_bg").fetch_all()['value']
        Q_1_bg = res_handles.get("Q1_bg").fetch_all()['value']
        I_2_bg = res_handles.get("I2_bg").fetch_all()['value']
        Q_2_bg = res_handles.get("Q2_bg").fetch_all()['value']



        # ss_thresh_1 = calib_params[0][0]
        # ss_factor_1 = calib_params[0][1]
        # ss_theta_1 = calib_params[0][2]
        # ss_thresh_2 = calib_params[1][0]
        # ss_factor_2 = calib_params[1][1]
        # ss_theta_2 = calib_params[1][2]


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

        IQ_1_bg = ss_factor_1 * np.real(np.exp(-1j * ss_theta_1) * (I_1_bg + 1j * Q_1_bg))
        IQ_2_bg = ss_factor_2 * np.real(np.exp(-1j * ss_theta_2) * (I_2_bg + 1j * Q_2_bg))
        IQ_10_1_bg = np.zeros(len(IQ_1_bg))
        IQ_10_2_bg = np.zeros(len(IQ_2_bg))



        for i, v in enumerate(IQ_1):
            if v > ss_thresh_1:
                IQ_10_1[i] = 1

        for i, v in enumerate(IQ_2):
            if v > ss_thresh_2: # maybe this sign is wrong
                IQ_10_2[i] = 1

        for i, v in enumerate(IQ_1_bg):
            if v > ss_thresh_1:
                IQ_10_1_bg[i] = 1

        for i, v in enumerate(IQ_2_bg):
            if v > ss_thresh_2: # maybe this sign is wrong
                IQ_10_2_bg[i] = 1

        ss_array_1 = IQ_10_1.reshape(shape)
        ss_array_2 = IQ_10_2.reshape(shape)

        ss_array_1_bg = IQ_10_1_bg.reshape(shape)
        ss_array_2_bg = IQ_10_2_bg.reshape(shape)

        ss_array = [ss_array_1,ss_array_2, ss_array_1_bg, ss_array_2_bg]


        return ss_array

    def job_to_shots_1q_3state(self, job, shape, calib_params, I_name='I', Q_name='Q'):

        I,Q = self.get_IQ(job, I_name, Q_name)

        IQ_0 = calib_params[0]
        IQ_1 = calib_params[1]
        IQ_2 = calib_params[2]

        IQ_210 = np.zeros(len(I),dtype=int)

        for i in range(len(I)):
            IQ_210[i] = find_state([I[i],Q[i]], IQ_0, IQ_1, IQ_2)

        ss_array = IQ_210.reshape(shape)

        return ss_array

    def ss_array_to_counts(self,ss_array,dim=3):

        ss_shape = np.shape(ss_array)

        counts_array = np.zeros([dim,ss_shape[1]])

        for i in range(ss_shape[0]):
            for j in range(ss_shape[1]):

                counts_array[ss_array[i][j]][j] += 1

        return counts_array


    def job_to_shots_2q(self, job, shape, calib_params):
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

        IQ_10 = np.zeros(len(IQ_1),dtype=int)

        for i in range(len(IQ_1)):
            IQ_10[i] = IQ_10_1[i]+2*IQ_10_2[i]

        ss_array = IQ_10.reshape(shape)
        counts = np.apply_along_axis(lambda x: np.bincount(x.ravel(), minlength=4), axis=0, arr=ss_array)
        return counts

    def job_to_IQ_avg_1q(self, job, shape, stream_tag_I="I", stream_tag_Q="Q"):
        """
        Return the list of all data [I,Q]
        """
        I_handle = job.result_handles.get(stream_tag_I)
        Q_handle = job.result_handles.get(stream_tag_Q)

        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()

        I = I_handle.fetch_all()['value']
        Q = Q_handle.fetch_all()['value']

        IQ_array = [I.reshape(shape),Q.reshape(shape)]

        return IQ_array

    def job_to_IQ_1q(self, job, shape):

        I_handle = job.result_handles.get("I")
        Q_handle = job.result_handles.get("Q")

        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()

        I = I_handle.fetch_all()['value']
        Q = Q_handle.fetch_all()['value']

        IQ_array = [I.reshape(shape),Q.reshape(shape)]

        return IQ_array

    def job_to_shots_arb(self, job, shape, discrim_class):

        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        I1 = res_handles.get("I1").fetch_all()['value']
        Q1 = res_handles.get("Q1").fetch_all()['value']
        I2 = res_handles.get("I2").fetch_all()['value']
        Q2 = res_handles.get("Q2").fetch_all()['value']
        Ic = res_handles.get("Ic").fetch_all()['value']
        Qc = res_handles.get("Qc").fetch_all()['value']

        IQ_dict = {'IQ1': I1 + 1j * Q1,
                   'IQ2': I2 + 1j * Q2,
                   'IQc': Ic + 1j * Qc
                    }


        state_array = np.zeros(len(IQ_dict['IQ1'])).astype('U')
        for i in range(len(IQ_dict['IQ1'])):
            IQ_point = {'IQ1': IQ_dict['IQ1'][i],
                        'IQ2': IQ_dict['IQ2'][i],
                        'IQc': IQ_dict['IQc'][i]
                        }
            state_array[i] = discrim_class.get_state(IQ_point)

        state_array_rs = state_array.reshape(shape)

        num_shots = shape[0]

        POVMs = discrim_class.POVM.keys()

        IQ_states = {key:np.zeros(shape[1]) for key in POVMs}

        for i, col in enumerate(state_array_rs.T):
            states, counts = np.unique(col, return_counts=True)
            for state,count in zip(states, counts):
                IQ_states[state][i] = count/num_shots
        return IQ_states

    def analyze(self, ydata, fit_func_name, seed_func_name, **kwargs):

        assert hasattr(fitters, fit_func_name), 'Invalid fit function name.'
        fit_func = getattr(fitters, fit_func_name)

        assert hasattr(fitters, seed_func_name), 'Invalid seed function name.'
        seed_func = getattr(fitters, seed_func_name)

        popt, pcov = sp.curve_fit(fit_func, self.data[0], ydata,
                                           seed_func(self.data, **kwargs))
        self.popt = popt
        self.pcov = pcov

        return popt, pcov

    def display(self, data=None, **kwargs):
        pass

    def save_data(self, data=None):

        if data is None:
            data=self.data

        with self.datafile() as f:
            for k, d in data.items():
                f.add(k, np.array(d))

    def load_data(self, f):
        data={}
        for k in f.keys():
            data[k]=np.array(f[k])
        data['attrs']=f.get_dict()
        return data

    def pickle_data(self):
        with open(self.pname, 'wb') as f:
            pickle.dump(self.data, f)

    def h5_data(self):
        with h5py.File(self.fname, "w") as f:
            for key, value in self.data.items():
                if isinstance(value, dict):
                    continue
                if isinstance(value, str):
                    f.create_dataset(key, data=value.encode("utf-8"))
                    continue
                try:
                    arr = np.asarray(value)
                except (ValueError, TypeError):
                    continue
                if arr.dtype.kind in ('O', 'U'):
                    continue
                f.create_dataset(key, data=arr)

