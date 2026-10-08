from qm.qua import *
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential, T1_seed
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class T2RRepeated(ExperimentClass):
    """
    T2R repeated experiment, no-phase cycling, for low-frequency noise spectroscopy
    """

    def __init__(self,element,tau_0, shots, calib_params,outer_folder='',path='T2R_Repeated', suffix='data', meta_dict=None, config=None,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        
        self.config=config
        self.element = element #meta dict element
        self.shots = shots
        self.tau_0 = tau_0 # fixed delay time
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "shots": self.shots,
            "tau_0": self.tau_0,
            'meta_dict': self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        tau_0 = self.tau_0
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)

            # declare stream
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):

                play('X90', qubit, timestamp_stream="timestamps")
                wait(tau_0 * u.ns, qubit)
                play('Y90', qubit)
                    
                # SS Readout
                align(qubit, resonator)
                I, Q = measure_1q(self.meta_dict, I, Q)
                assign(I_reset, I)
                with if_(I > ss_thresh):
                    assign(state, 1)
                with else_():
                    assign(state, 0)

                save(state, ss_stream)

                align(qubit, resonator)
            
                # Active reset
                with while_(I_reset > ground_confidence_threshold):
                    with if_(I_reset > ss_thresh):
                        play('X180', qubit)
                        align(qubit, resonator)
                    I, _ = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                ss_stream.save_all("ss")

        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["timestamps", "ss"])
        results = result.fetch_all()
        timestamps = results[0]
        # on an opx, returned timestamps had a different shape (not flat)
        timestamps = np.array([t[0] for t in timestamps])
        ss_array = results[1]

        self.data.update(
            {'ss': ss_array, 
             'timestamps': np.diff(timestamps, prepend=0), # clock cycles, difference array
            }
        )
