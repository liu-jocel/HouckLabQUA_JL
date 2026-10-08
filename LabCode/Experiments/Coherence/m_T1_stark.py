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

class ACStarkT1(ExperimentClass):
    """
    AC Stark T1 experiment
    """
    def __init__(self,element,t_vec, shots, stark_detuning, stark_amp, calib_params,outer_folder='',path='T1', suffix='data', meta_dict=None, config=None, num_pulses=1,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config=config
        val = self.config["waveforms"]["ps_wf_q2"]["sample"]
        assert isinstance(val, (float, int)), f"Invalid sample type: {type(val)}"
        self.num_pulses = num_pulses
        self.element = element #meta dict element
        self.t_vec = t_vec # units clock cycles
        self.shots = shots
        self.stark_detuning = stark_detuning
        self.stark_amp = stark_amp
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
        qubit_freq = self.meta_dict['IF_freq']
        stark_detuned_freq = self.meta_dict['IF_freq'] + self.stark_detuning
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            m = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    update_frequency(qubit, qubit_freq)
                    play('X90', qubit)
                    play('X90', qubit)

                    align(qubit, resonator)
                    update_frequency(qubit, stark_detuned_freq)
                    play("constant_tone", qubit, duration=t)
                    align(qubit, resonator)
                    update_frequency(qubit, qubit_freq)
                    wait(1 * u.us, qubit)

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
                    wait(resonator_dump_time*u.ns, qubit)
                # wait(2000 * u.us, qubit)

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
            ax.set_title(f"{self.element} Stark Amp {self.stark_amp} Detuning {self.stark_detuning / 1e6} MHz \n T1 = {str(np.round(T1, 4))} +/-{str(np.round(T1_err, 4))} us")
        except:
            ax.set_title(f"{self.element} num Pulses: {self.num_pulses}.")
            print("Fit Failed")
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Excited state probability')
        plt.semilogx()
        plt.savefig(self.iname[:-4] + 'ss' + '.png', bbox_inches="tight")
        plt.close(fig)