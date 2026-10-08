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

class T2RQuadRepeated(ExperimentClass):
    """
    T2R quadrature experiment, for low-frequency noise spectroscopy
    """

    def __init__(self,element,tau_0, shots, calib_params,outer_folder='',path='T2R_Quad', suffix='data', meta_dict=None, config=None,):
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
            m = declare(int) # index for phase cycling, 0 -> +x, 1 -> -x, 2 -> +y, 3 -> -y
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)

            # declare streams for each of the 4 phase cycles
            ss_stream_plus_x = declare_stream()
            ss_stream_minus_x = declare_stream()
            ss_stream_plus_y = declare_stream()
            ss_stream_minus_y = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(m, 0, m < 4, m + 1):

                    play('X90', qubit, timestamp_stream="timestamps")
                    wait(tau_0 * u.ns, qubit)
                
                    with if_(m == 0):
                        # +X
                        play('X90', qubit)
                    with elif_(m == 1):
                        # -X
                        play('mX90', qubit)
                    with elif_(m == 2):
                        # +Y
                        play('Y90', qubit)
                    with else_():
                        # -Y
                        play('mY90', qubit)

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)

                    # NOTE I was forced to do this multiple nesting because for some reason, elif_() did not work here (resulted in program hanging)
                    with if_(m == 0):
                        save(state, ss_stream_plus_x)
                    with else_():
                        with if_(m == 1):
                            save(state, ss_stream_minus_x)
                        with else_():
                            with if_(m == 2):
                                save(state, ss_stream_plus_y)
                            with else_():
                                save(state, ss_stream_minus_y)

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
                ss_stream_plus_x.save_all("ss_+x")
                ss_stream_minus_x.save_all("ss_-x")
                ss_stream_plus_y.save_all("ss_+y")
                ss_stream_minus_y.save_all("ss_-y")

        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["timestamps", "ss_+x", "ss_-x", "ss_+y", "ss_-y"])
        results = result.fetch_all()
        timestamps = results[0]
        # on an opx, returned timestamps had a different shape (not flat)
        #timestamps = np.array([t[0] for t in timestamps])
        ss_plus_x_array = results[1]
        ss_minus_x_array = results[2]
        ss_plus_y_array = results[3]
        ss_minus_y_array = results[4]

        self.data.update(
            {'ss_plus_x': ss_plus_x_array, 
             'ss_minus_x': ss_minus_x_array, 
             'ss_plus_y': ss_plus_y_array,
             'ss_minus_y': ss_minus_y_array,
             'timestamps': np.diff(timestamps, prepend=0).astype(np.int32), # clock cycles, difference array
            }
        )
