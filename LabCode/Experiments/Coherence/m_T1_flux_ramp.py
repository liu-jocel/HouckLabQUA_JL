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

class T1FluxRamp(ExperimentClass):
    """
    T1 experiment
    """
    def __init__(self,element, t_vec, dc_excursion, shots, calib_params,outer_folder='',path='T1', suffix='data', meta_dict=None, config=None,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config=config
        self.dc_excursion=dc_excursion
        self.element = element #meta dict element
        self.t_vec = t_vec # units clock cycles
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            'dc_excursion': self.dc_excursion,
            "t_vec": self.t_vec,
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
        flux_line = self.meta_dict['flux_name']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        idle_voltage = self.meta_dict['flux_dc_offset']

        with program() as prog:
            n = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    align(qubit, resonator, flux_line)
                    play('X90', qubit)
                    play('X90', qubit)
                    align(qubit, resonator, flux_line)
                    set_dc_offset(flux_line, 'single', self.dc_excursion)
                    align(qubit, resonator, flux_line)
                    wait(t, qubit, resonator, flux_line)
                    align(qubit, resonator, flux_line)
                    set_dc_offset(flux_line, 'single', idle_voltage)
                    # SS Readout
                    align(qubit, resonator, flux_line)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    align(qubit, resonator, flux_line)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream)
                    align(qubit, resonator, flux_line)

                    # # Active reset
                    # with while_(I_reset > ground_confidence_threshold):
                    #     with if_(I_reset > ss_thresh):
                    #         play('X180', qubit)
                    #         align(qubit, resonator, flux_line)
                    #     I, _ = measure_1q(self.meta_dict, I, Q)
                    #     assign(I_reset, I)
                    # align(qubit, resonator, flux_line)
                    wait(reset_time*u.ns, qubit, resonator, flux_line)

            with stream_processing():
                ss_stream.buffer(len(self.t_vec)).average().save("ss")

        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["ss"])
        ss_array = result.fetch_all()[0]
        t_vec = self.t_vec / u.us
        self.data.update({'ss_data': ss_array})

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(t_vec, ss_array, 'o')
        try:
            seed = T1_seed(t_vec, ss_array)
            popt, pcov = curve_fit(exponential, t_vec, ss_array, p0=seed)

            T1 = popt[2]
            T1_err = np.sqrt(pcov[2][2])
            self.data.update({
                'T1': (T1, T1_err),
                'fit_popt': popt, 'fit_pcov': pcov,
            })

            t_vec_fit = np.logspace(np.log10(t_vec[0]), np.log10(t_vec[-1]))
            ax.plot(t_vec_fit, exponential(t_vec_fit, *popt))
            ax.set_title(f"{self.element}. DC point: {self.dc_excursion}. T1 = {str(np.round(T1, 4))} +/-{str(np.round(T1_err, 4))} us")
        except:
            ax.set_title(f"{self.element}. DC point: {self.dc_excursion}.")
            print("Fit Failed")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Excited state probability')
        plt.semilogx()
        plt.savefig(self.iname[:-4] + 'ss' + '.png', bbox_inches="tight")
        plt.close(fig)

