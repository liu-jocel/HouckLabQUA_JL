from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
from datetime import datetime
import os
import sys
sys.path.append('/')
import numpy as np
import matplotlib
matplotlib.use("qtagg", force=True)
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Config.make_meta_dict_flux_tunable import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q_ft
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Helpers.Shot_Analysis.shot_analysis import SingleShotAnalysis

def cosine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset



class AdiabaticRamp(ExperimentClass):
    """
    Qubit spectroscopy versus flux
    """
    def __init__(self, element, dc_excursion, shots, repeats = 1, outer_folder='',path='trans', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.dc_excursion = dc_excursion
        self.shots = shots
        self.repeats = repeats
        self.config = config
        self.job, qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.data = {
            'qubit': self.element,
            'dc_excursion': self.dc_excursion,
            'shots': self.shots,
            "meta_dict": self.meta_dict
        }

        # Analyze and save data
        job, qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(job, qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        idle_voltage = self.meta_dict['flux_dc_offset']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        flux_line = self.meta_dict['flux_name']
        flux_settle_time = self.meta_dict['flux_settle_time']
        resonator = self.meta_dict['r_name']

        with program() as prog:
            n = declare(int)  # QUA variable for the averaging index
            m = declare(int)    # QUA variable for repeating the flux ramp
            I = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            I_idle_ground = declare_stream()  # Stream for the 'I' quadrature
            Q_idle_ground = declare_stream()  # Stream for the 'Q' quadrature
            I_idle_excited = declare_stream()  # Stream for the 'I' quadrature
            Q_idle_excited = declare_stream()  # Stream for the 'Q' quadrature
            I_ramp_ground = declare_stream()  # Stream for the 'I' quadrature
            Q_ramp_ground = declare_stream()  # Stream for the 'Q' quadrature
            I_ramp_excited = declare_stream()  # Stream for the 'I' quadrature
            Q_ramp_excited = declare_stream()  # Stream for the 'Q' quadrature
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(n, 0, n < shots, n + 1):
                wait(reset_time * u.ns, flux_line, resonator, qubit)
                align(qubit, resonator, flux_line)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_idle_ground)
                save(Q, Q_idle_ground)

                wait(reset_time * u.ns, flux_line, resonator, qubit)
                align(qubit, resonator, flux_line)
                play('X180', qubit)
                align(qubit, resonator, flux_line)
                with for_(m, 0, m < self.repeats, m + 1):
                    wait(1*u.ns, qubit, resonator, flux_line)
                    wait(1*u.ns, qubit, resonator, flux_line)
                align(qubit, resonator, flux_line)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_idle_excited)
                save(Q, Q_idle_excited)

                wait(reset_time * u.ns, flux_line, resonator, qubit)
                align(qubit, resonator, flux_line)
                with for_(m, 0, m < self.repeats, m + 1):
                    set_dc_offset(flux_line, 'single', self.dc_excursion)
                    set_dc_offset(flux_line, 'single', idle_voltage)
                align(qubit, resonator, flux_line)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_ramp_ground)
                save(Q, Q_ramp_ground)

                wait(reset_time * u.ns, flux_line, resonator, qubit)
                align(qubit, resonator, flux_line)
                play('X180', qubit)
                align(qubit, resonator, flux_line)
                with for_(m, 0, m < self.repeats, m + 1):
                    set_dc_offset(flux_line, 'single', self.dc_excursion)
                    set_dc_offset(flux_line, 'single', idle_voltage)
                align(qubit, resonator, flux_line)
                I, Q = measure_1q(self.meta_dict, I, Q)
                save(I, I_ramp_excited)
                save(Q, Q_ramp_excited)
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
                I_idle_ground.buffer(1000).flatten().save_all("I_idle_ground")
                Q_idle_ground.buffer(1000).flatten().save_all("Q_idle_ground")
                I_idle_excited.buffer(1000).flatten().save_all("I_idle_excited")
                Q_idle_excited.buffer(1000).flatten().save_all("Q_idle_excited")
                I_ramp_ground.buffer(1000).flatten().save_all("I_ramp_ground")
                Q_ramp_ground.buffer(1000).flatten().save_all("Q_ramp_ground")
                I_ramp_excited.buffer(1000).flatten().save_all("I_ramp_excited")
                Q_ramp_excited.buffer(1000).flatten().save_all("Q_ramp_excited")
                n_st.save("iteration")
        return prog

    def analyze(self, job, qm):
        data_list = ['I_idle_ground', 'Q_idle_ground', 'I_idle_excited', 'Q_idle_excited',
                     'I_ramp_ground', 'Q_ramp_ground', 'I_ramp_excited', 'Q_ramp_excited',
                     'iteration']
        results = fetching_tool(job, data_list=data_list, mode="live")


        def flatten_and_match(a, b):
            """Flatten and truncate both arrays to the same length."""
            a = np.ravel(a)
            b = np.ravel(b)
            min_len = min(len(a), len(b))
            return a[:min_len], b[:min_len]

        # Initialize storage lists
        I_ig, Q_ig = [], []  # Idle ground
        I_ie, Q_ie = [], []  # Idle excited
        I_rg, Q_rg = [], []  # Ramp ground
        I_re, Q_re = [], []  # Ramp excited

        # --- Create 3x1 subplot grid ---
        plt.ion()
        fig, (ax_ground, ax_excited, ax_all) = plt.subplots(3, 1, figsize=(8, 12))
        fig.suptitle(f"Adiabatic Flux Ramp: {self.repeats} Ramps")
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

        # Scatter placeholders
        scat_ig = ax_ground.scatter([], [], alpha=0.4, s=15, marker='o', color='tab:blue', label="Idle Ground")
        scat_rg = ax_ground.scatter([], [], alpha=0.4, s=15, marker='^', color='tab:green', label="Ramp Ground")

        scat_ie = ax_excited.scatter([], [], alpha=0.4, s=15, marker='s', color='tab:orange', label="Idle Excited")
        scat_re = ax_excited.scatter([], [], alpha=0.4, s=15, marker='x', color='tab:red', label="Ramp Excited")

        scat_all_ig = ax_all.scatter([], [], alpha=0.4, s=15, marker='o', color='tab:blue', label="Idle Ground")
        scat_all_rg = ax_all.scatter([], [], alpha=0.4, s=15, marker='^', color='tab:green', label="Ramp Ground")
        scat_all_ie = ax_all.scatter([], [], alpha=0.4, s=15, marker='s', color='tab:orange', label="Idle Excited")
        scat_all_re = ax_all.scatter([], [], alpha=0.4, s=15, marker='x', color='tab:red', label="Ramp Excited")

        # Titles and axis setup
        ax_ground.set_title("Ground: Idle vs Ramp")
        ax_excited.set_title("Excited: Idle vs Ramp")
        ax_all.set_title("All Data: Ground + Excited")

        for ax in [ax_ground, ax_excited, ax_all]:
            ax.set_xlabel("I (a.u.)")
            ax.set_ylabel("Q (a.u.)")
            ax.grid(True)
            ax.legend(loc="upper right")

        plt.tight_layout()
        # --- Live update loop ---
        while results.is_processing():
            (
                I_idle_ground, Q_idle_ground,
                I_idle_excited, Q_idle_excited,
                I_ramp_ground, Q_ramp_ground,
                I_ramp_excited, Q_ramp_excited,
                iteration
            ) = results.fetch_all()

            progress_counter(
                iteration, self.shots,
                progress_bar=True,
                percent=True,
                start_time=results.get_start_time()
            )

            # Flatten and match
            I_idle_ground, Q_idle_ground = flatten_and_match(I_idle_ground, Q_idle_ground)
            I_idle_excited, Q_idle_excited = flatten_and_match(I_idle_excited, Q_idle_excited)
            I_ramp_ground, Q_ramp_ground = flatten_and_match(I_ramp_ground, Q_ramp_ground)
            I_ramp_excited, Q_ramp_excited = flatten_and_match(I_ramp_excited, Q_ramp_excited)

            # Append
            I_ig.extend(I_idle_ground)
            Q_ig.extend(Q_idle_ground)
            I_ie.extend(I_idle_excited)
            Q_ie.extend(Q_idle_excited)
            I_rg.extend(I_ramp_ground)
            Q_rg.extend(Q_ramp_ground)
            I_re.extend(I_ramp_excited)
            Q_re.extend(Q_ramp_excited)

            # Update scatter data
            scat_ig.set_offsets(np.column_stack((I_ig, Q_ig)))
            scat_rg.set_offsets(np.column_stack((I_rg, Q_rg)))
            scat_ie.set_offsets(np.column_stack((I_ie, Q_ie)))
            scat_re.set_offsets(np.column_stack((I_re, Q_re)))

            scat_all_ig.set_offsets(np.column_stack((I_ig, Q_ig)))
            scat_all_rg.set_offsets(np.column_stack((I_rg, Q_rg)))
            scat_all_ie.set_offsets(np.column_stack((I_ie, Q_ie)))
            scat_all_re.set_offsets(np.column_stack((I_re, Q_re)))

            # Auto-resize axes around data
            all_I = np.concatenate([I_ig, I_ie, I_rg, I_re]) if I_ig else [0]
            all_Q = np.concatenate([Q_ig, Q_ie, Q_rg, Q_re]) if Q_ig else [0]
            for ax in [ax_ground, ax_excited, ax_all]:
                ax.set_xlim(np.min(all_I), np.max(all_I))
                ax.set_ylim(np.min(all_Q), np.max(all_Q))
            plt.pause(0.5)

        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        qm.close()

        # Gaussian Fitting Data
        I_calibrate = np.array(I_ig + I_ie)
        Q_calibrate = np.array(Q_ig + Q_ie)
        I_ig = np.array(I_ig)
        Q_ig = np.array(Q_ig)
        I_ie = np.array(I_ie)
        Q_ie = np.array(Q_ie)
        I_rg = np.array(I_rg)
        Q_rg = np.array(Q_rg)
        I_re = np.array(I_re)
        Q_re = np.array(Q_re)
        qubit_freq_mat = np.array([[0, 4000], [-4000, 0]])
        outerFolder = os.path.join(self.outerFolder, self.path, self.path + "_" + self.date_string)

        ss_calibrate = SingleShotAnalysis(I_calibrate, Q_calibrate, cen_num=2, outerFolder=outerFolder,
                                           name=self.time_string + f"_calibrate", fast=True,
                                           disp_image=False, qubit_freq_mat=qubit_freq_mat)
        blob_centers = ss_calibrate.get_Centers(I_calibrate, Q_calibrate)

        analysis_ig = SingleShotAnalysis(I_ig, Q_ig, cen_num=2, outerFolder=outerFolder,
                                           name=self.time_string + f"_idle_ground_{self.repeats}", fast=True,
                                           disp_image=True, qubit_freq_mat=qubit_freq_mat, i_0_arr = I_calibrate, q_0_arr = Q_calibrate,
                                         centers=blob_centers)
        ig_pops = analysis_ig.estimate_populations()
        self.data.update(ig_pops)
        print(f"IG population: {ig_pops}")
        analysis_ie = SingleShotAnalysis(I_ie, Q_ie, cen_num=2, outerFolder=outerFolder,
                                           name=self.time_string + f"_idle_excited_{self.repeats}", fast=True,
                                           disp_image=True, qubit_freq_mat=qubit_freq_mat, i_0_arr = I_calibrate, q_0_arr = Q_calibrate,
                                         centers=blob_centers)
        ie_pops = analysis_ie.estimate_populations()
        self.data.update(ie_pops)
        print(f'IE population: {ie_pops}')
        analysis_rg = SingleShotAnalysis(I_rg, Q_rg, cen_num=2, outerFolder=outerFolder,
                                           name=self.time_string + f"_ramp_ground_{self.repeats}", fast=True,
                                           disp_image=True, qubit_freq_mat=qubit_freq_mat, i_0_arr = I_calibrate, q_0_arr = Q_calibrate,
                                         centers=blob_centers)
        rg_pops = analysis_rg.estimate_populations()
        self.data.update(rg_pops)
        print(f"RG population: {rg_pops}")
        analysis_re = SingleShotAnalysis(I_re, Q_re, cen_num=2, outerFolder=outerFolder,
                                           name=self.time_string + f"_ramp_excited_{self.repeats}", fast=True,
                                           disp_image=True, qubit_freq_mat=qubit_freq_mat, i_0_arr = I_calibrate, q_0_arr = Q_calibrate,
                                         centers=blob_centers)
        re_pops = analysis_re.estimate_populations()
        self.data.update(re_pops)
        print(f"RE population: {re_pops}")

        IQ_idle_ground = I_idle_ground + 1j*Q_idle_ground
        IQ_idle_excited = I_idle_excited + 1j*Q_idle_excited
        IQ_ramp_ground = I_ramp_ground + 1j*Q_ramp_ground
        IQ_ramp_excited = I_ramp_excited + 1j*Q_ramp_excited

        ground_pop_idx = np.argmax(ig_pops['mean'])
        excited_pop_idx = np.argmax(ie_pops['mean'])
        assert ground_pop_idx != excited_pop_idx
        # Measurement of the fidelity between idling and ramping the state
        ground_overlap = 1/(1+np.abs(np.log(ig_pops['mean'][ground_pop_idx]/rg_pops['mean'][ground_pop_idx])))
        print(ground_overlap)

        # Excited state overlap
        excited_overlap = 1/(1+np.abs(np.log(ie_pops['mean'][excited_pop_idx]/re_pops['mean'][excited_pop_idx])))
        print(excited_overlap)


        self.data.update({
            "IQ_idle_ground": IQ_idle_ground,
            "IQ_idle_excited": IQ_idle_excited,
            "IQ_ramp_ground": IQ_ramp_ground,
            "IQ_ramp_excited": IQ_ramp_excited,
            'ramp_vs_idle_ground': ground_overlap,
            'ramp_vs_idle_excited': excited_overlap
        })



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])


if __name__ == "__main__":
    qubit ='q1'
    meta_dict = make_meta_dict(qubit)[qubit]
    config = meta_dict_to_config_1Q_ft(meta_dict)
    LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
    print(LOs)
    # set_LO_parameters(LOs)
    ramp_center = meta_dict['flux_dc_offset']
    dc_span = 0.2
    num_points = 1
    dc_vec = np.linspace(ramp_center-dc_span/2, ramp_center+dc_span/2, num_points)
    g_overlap = []
    e_overlap = []
    for dc in dc_vec:
        AR = AdiabaticRamp(qubit, 0.1, 20000, repeats=1, meta_dict=meta_dict, outer_folder=meta_dict['global_params']['outer_folder'],
                        path=qubit, suffix="Adiabatic_Ramp_Test", config=config)
        g_overlap.append(AR.data['ramp_vs_idle_ground'])
        e_overlap.append(AR.data['ramp_vs_idle_excited'])

    plt.plot(dc_vec, g_overlap, label="ground_state")
    plt.plot(dc_vec, e_overlap, label="excited_state")
    plt.title("Ramp Voltage vs State Fidelity")
    plt.xlabel("Ramp Voltage (V)")
    plt.xlabel("exp[-|log(p_idle/p_ramp)|")
    plt.show()