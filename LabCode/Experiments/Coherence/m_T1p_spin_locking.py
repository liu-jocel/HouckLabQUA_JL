from qm.qua import *
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential, T1_seed
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class T1pSpinLocking(ExperimentClass):
    """
    T1p Spin Locking experiment, SL3 with phase cycling
    """
    def __init__(self,element,t_vec, sl_amp, shots,calib_params,outer_folder='',path='T1p', suffix='data', meta_dict=None, config=None,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config=config
        self.element = element #meta dict element
        self.t_vec = t_vec # units clock cycles
        self.sl_amp = sl_amp # spin-locking pulse amplitude
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "t_vec": self.t_vec,
            "sl_amp": self.sl_amp,
            'shots': self.shots,
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
        sl_amp = self.sl_amp
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            m = declare(int) # index for phase cycling, 0 -> +y, 1 -> -y
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
    
            # declare streams for both +y and -y phase cycling, and for the averaged result
            ss_stream_plus_y = declare_stream()
            ss_stream_minus_y = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    with for_(m, 0, m < 2, m + 1):

                        with if_(m == 0):
                            # SL3_+y
                            play('Y90', qubit)
                            play('constant_tone' * amp(sl_amp), qubit, duration=t)
                            play('Y90', qubit)
                        with else_():
                            # SL3_-y
                            play("mY90", qubit)
                            play('constant_tone' * amp(sl_amp), qubit, duration=t)
                            play("mY90", qubit) 

                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)

                        with if_(m == 0):
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
                ss_stream_plus_y.buffer(len(self.t_vec)).average().save("ss_+y")
                ss_stream_minus_y.buffer(len(self.t_vec)).average().save("ss_-y")
                # Average the +y and -y results to mitigate effects of state preparation and measurement errors
                (ss_stream_plus_y + ss_stream_minus_y).multiply_by(0.5).buffer(len(self.t_vec)).average().save("ss")

        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss", "ss_+y", "ss_-y"])
        results = result.fetch_all()
        ss_array = results[0]
        ss_plus_y_array = results[1]
        ss_minus_y_array = results[2]
      
        t_vec = self.t_vec / u.us
        self.data.update({'ss_data': ss_array, 'ss_plus_y': ss_plus_y_array, 'ss_minus_y': ss_minus_y_array})

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.scatter(t_vec, ss_array, s=24, c="r", label="avg")
        ax.scatter(t_vec, ss_plus_y_array, s=24, c="0.25", label="+y")
        ax.scatter(t_vec, ss_minus_y_array, s=24, c="0.75", label="-y")

        try:
            seed = T1_seed(t_vec, ss_array)
            popt, pcov = curve_fit(exponential, t_vec, ss_array, p0=seed)

            T1p = popt[2]
            T1p_err = np.sqrt(pcov[2][2])
            self.data.update({
                'T1p': (T1p, T1p_err),
                'fit': (popt, pcov),
            })

            t_vec_fit = np.logspace(np.log10(t_vec[0]), np.log10(t_vec[-1]))
            ax.plot(t_vec_fit, exponential(t_vec_fit, *popt))
            ax.set_title(f"{self.element} Spin Locking sl_amp: {self.sl_amp:.2f}. T1p = {str(np.round(T1p, 4))} +/-{str(np.round(T1p_err, 4))} us")
        except:
            ax.set_title(f"{self.element} Spin Locking sl_amp: {self.sl_amp:.2f}.")
            print("Fit Failed")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Excited state probability')
        plt.semilogx()
        plt.savefig(self.iname[:-4] + 'ss' + '.png', bbox_inches="tight")
        plt.close(fig)
