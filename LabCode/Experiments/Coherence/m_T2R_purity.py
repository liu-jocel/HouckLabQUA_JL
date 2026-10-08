from qm.qua import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.fit_functions import exponential, T2E_seed
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class T2R_Purity(ExperimentClass):
    """
    T2R experiment. On resonance driving,
    Measure expectation values of X and Y to extract purely decay envelope.
    """
    def __init__(self, element, t_vec, shots, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.t_vec = t_vec*u.ns
        self.tau_vec_normalized_clk = np.array([int(t/(8)) for t in t_vec], dtype=int) # 4 ns in a clock cycle, minimun wait is t_vec/2
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
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
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream_x = declare_stream()
            ss_stream_y = declare_stream()
            ss_stream_z = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(t, self.tau_vec_normalized_clk):
                    play('X90', qubit)
                    wait(t, qubit)
                    play('X90', qubit)
                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream_x)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)

                    # Experiment 2: Measure along Y
                    play('X90', qubit)
                    wait(t, qubit)
                    play('Y90', qubit)

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream_y)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)

                    # Experiment 3: Measure along Z
                    play('X90', qubit)
                    wait(t, qubit)

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, ss_stream_z)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play('X180', qubit)
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time*u.ns, qubit)

            with stream_processing():
                ss_stream_x.buffer(len(self.t_vec)).average().save("population_x")
                ss_stream_y.buffer(len(self.t_vec)).average().save("population_y")
                ss_stream_y.buffer(len(self.t_vec)).average().save("population_z")
        return prog

    def analyze(self):
        result = fetching_tool(self.job, data_list=["population_x", 'population_y', 'population_z'])
        population_x, population_y, population_z = result.fetch_all()
        self.data.update({'population_x': population_x,
                          'population_y': population_y,
                          'population_y': population_z})
        t_vec = self.t_vec/u.us

        # Calculate expectation values and equatorial purity
        sigma_x = 1 - 2*population_x
        sigma_y = 1 - 2*population_y
        sigma_z = 1 - 2*population_z
        equatorial_purity = 0.5*(sigma_x**2 + sigma_y**2 + sigma_z**2)

        # Plot data
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(t_vec, equatorial_purity, 'o')
        try:
            seed = T2E_seed(t_vec, equatorial_purity)
            popt, pcov = curve_fit(exponential, t_vec, equatorial_purity, p0=seed, maxfev=5000)

            T2 = popt[2]
            T2_err = np.sqrt(pcov[2][2])
            ax.plot(t_vec, exponential(t_vec, *popt))
            ax.set_title(f'{self.element}, T2R =  {str(np.round(T2, 4))} +/- {str(np.round(T2_err, 4))} us')

            update_dict = {'T2R': (T2, T2_err), 'fit_popt': popt, 'fit_pcov': pcov}
            self.data.update(update_dict)

        except:
            print("Fit Failed")
            ax.set_title(self.element+', T2R')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Excited state probability')
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)