"""
Bootstrap Tomography
Characterizing imperfect pulses following Dobrovitski et al.
"""

from qm.qua import *
import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.Shot_Analysis.post_selection_SSA import PostSelection, gaussian_2d
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

# Default bootstrap tomography sequences (Dobrovitski et al.)
# Pulse names must match operations defined in the config.
DEFAULT_SEQUENCE = [
    ("X90",),
    ("Y90",),
    ("X180", "X90"),
    ("Y180", "Y90"),
    ("X90", "Y180"),
    ("Y90", "X180"),
    ("X90", "Y90"),
    ("Y90", "X90"),
    ("Y90", "X180", "X90"),
    ("X90", "X180", "Y90"),
    ("Y90", "Y180", "X90"),
    ("X90", "Y180", "Y90" ),
]


def _extract_errors(populations_1d, calib_params):
    """Extract bootstrap error parameters from a 12-element populations array."""
    confusion_matrix = np.array([[1,0],[0,1]]) # no confusion matrix np.array(calib_params['confusion'])
    c10 = confusion_matrix[1, 0]
    c11 = confusion_matrix[1, 1]
    populations = (populations_1d - c10) / (c11 - c10)

    szs = 2 * populations - 1
    phi_prime = -szs[0] / 2
    chi_prime = -szs[1] / 2
    phi = szs[2] / 2 - phi_prime
    chi = szs[3] / 2 - chi_prime
    vz = (2 * phi_prime - szs[4]) / 2
    ez = (szs[5] - 2 * chi_prime) / 2

    A = np.array([
        [-1, -1, -1, 0, 0],
        [+1, -1, +1, 0, 0],
        [+1, +1, -1, 2, 0],
        [-1, +1, +1, 2, 0],
        [-1, -1, +1, 0, 2],
        [+1, -1, -1, 0, 2],
    ], dtype=float)
    b = szs[6:12]
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    ez_prime, vx_prime, vz_prime, ey, vx = x
    ey_prime = 0.0

    return {
        'phi_prime': phi_prime, 'chi_prime': chi_prime,
        'phi': phi, 'chi': chi,
        'ez': ez, 'vz': vz,
        'ey_prime': ey_prime, 'ez_prime': ez_prime,
        'vx_prime': vx_prime, 'vz_prime': vz_prime,
        'ey': ey, 'vx': vx,
        '_A': A, '_sv': sv, '_populations': populations,
    }


class BootstrapTomography(ExperimentClass):
    """
    Bootstrap tomography experiment for characterizing imperfect pulses.
    Following Dobrovitski et al.
    """

    def __init__(self, element, shots, calib_params, sequence=None, outer_folder='',
                 path='BootstrapTomography', suffix='data', meta_dict=None, config=None,
                 simulate=False, sim_duration=10_000):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)

        self.element = element
        self.shots = shots
        self.calib_params = calib_params
        self.sequence = sequence if sequence is not None else DEFAULT_SEQUENCE
        self.config = config

        self.data = {
            'qubit': self.element,
            'shots': self.shots,
            'sequence': [list(s) for s in self.sequence],
            'meta_dict': self.meta_dict,
        }

        if simulate:
            self._simulate_and_save(self.make_prog(), self.config, duration=sim_duration)
        else:
            self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
            self.analyze(self.job, self.qm)
            self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            self.h5_data()

    def _simulate_and_save(self, prog, config, duration=10_000):
        """Run waveform simulation and save the report alongside the experiment image."""
        from qm import SimulationConfig
        from LabCode.CoreLib.experiment import qmm
        simulation_config = SimulationConfig(duration=duration)
        qm = qmm.open_qm(config)
        job = qm.simulate(prog, simulation_config)
        samples = job.get_simulated_samples()
        waveform_report = job.get_simulated_waveform_report()
        waveform_report.create_plot(samples, plot=True, save_path=self.iname)
        return waveform_report.to_dict(), samples, job

    def _play_sequence(self, pulses, qubit):
        """Play a tuple of pulse names on the qubit. Called at QUA compile time."""
        if isinstance(pulses, str):
            pulses = (pulses,)
        for pulse in pulses:
            play(pulse, qubit)

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            state = declare(int)
            I_st = declare_stream()
            Q_st = declare_stream()
            state_st = declare_stream()
            n_st = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                for i in range(len(self.sequence)):
                    reset_frame(qubit)
                    align()
                    with strict_timing_():
                        self._play_sequence(self.sequence[i], qubit)
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(I, I_st)
                    save(Q, Q_st)
                    save(state, state_st)
                    wait(resonator_dump_time * u.ns, qubit)
                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.sequence)).average().save("I")
                Q_st.buffer(len(self.sequence)).average().save("Q")
                I_st.buffer(len(self.sequence)).save_all("I_raw")
                Q_st.buffer(len(self.sequence)).save_all("Q_raw")
                state_st.buffer(len(self.sequence)).average().save("populations")
                n_st.save("iteration")

        return prog

    def analyze(self, job, qm):

        bs_ss_dir = os.path.join(os.path.dirname(self.iname), "BS_SS")
        os.makedirs(bs_ss_dir, exist_ok=True)

        results = fetching_tool(job, data_list=['populations', "I", "Q",], mode="live")
        x_labels = [" ".join(pulses) for pulses in self.sequence]
        x = range(len(self.sequence))
        self.data['sequence'] = x_labels

        matplotlib.use("qtagg", force=True)
        # Live Plotting
        fig, ax = plt.subplots()
        hline = ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="0.5 Reference")
        (line,) = ax.plot(x, [0.5]*len(self.sequence), "o-", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Sequence")
        ax.set_ylabel("Population")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Bootstrap Tomography pulses")
        interrupt_on_close(fig, job)

        populations_raw = np.full(len(self.sequence), 0.5)
        while results.is_processing():
            populations_raw, I, Q = results.fetch_all()
            self.data['I'] = I
            self.data['Q'] = Q

            # err = _extract_errors(populations_raw, self.calib_params)
            # populations = err['_populations']
            # self.data['populations'] = populations

            # line.set_data(x, populations)
            # ax.relim()
            # ax.autoscale_view(scaley=False)
            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.pause(0.05)

        I_raw = job.result_handles.get("I_raw").fetch_all()['value']  # [shots, n_seq]
        Q_raw = job.result_handles.get("Q_raw").fetch_all()['value']  # [shots, n_seq]
        self.data['I_raw'] = I_raw
        self.data['Q_raw'] = Q_raw
        qm.close()
        plt.ioff()

        

        # fig.savefig(self.iname, bbox_inches="tight")
        # plt.close(fig)
        # self.data['populations'] = populations_raw

        # Post-selection for each sequence
        ps_populations = []
        for idx, seq in enumerate(self.sequence):
            I_seq = I_raw[:, idx].astype(float)*1e5
            Q_seq = Q_raw[:, idx].astype(float)*1e5
            ps = PostSelection(I_seq, Q_seq, i_0_arr=I_seq, q_0_arr=Q_seq)
            fit_results, centers, _, selected_data_arr, populations = ps.do_post_select(display=False)
            ps_populations.append(populations)
            seq_label = "_".join(seq)
            savepath = os.path.join(bs_ss_dir, seq_label + ".png")
            ps.display_ps_confusion(fit_results, selected_data_arr, centers, populations=populations, save=True, savepath=savepath)
        self.data['ps_populations'] = np.array(ps_populations)

        # Plot post-selected excited state probability vs sequence
        ps_excited = [pop[1] for pop in ps_populations]
        fig_ps, ax_ps = plt.subplots(constrained_layout=True, figsize=(10, 4))
        ax_ps.axhline(0.5, color="gray", linestyle="--", lw=1, label="0.5 Reference")
        ax_ps.plot(x, ps_excited, "o-", lw=2)
        ax_ps.set_xticks(list(x))
        ax_ps.set_xticklabels(x_labels, rotation=45, ha="right")
        ax_ps.set_xlabel("Sequence")
        ax_ps.set_ylabel("P(excited) — post-selected")
        ax_ps.set_ylim(0.5 - 0.2, 0.5 + 0.2)
        ax_ps.grid()
        ax_ps.set_title(f"Bootstrap Tomography — post-selected populations")
        root, ext = os.path.splitext(self.iname)
        fig_ps.savefig(f"{root}_ps{ext}", bbox_inches="tight")
        plt.close(fig_ps)

        ### Analysis
        err = _extract_errors(ps_excited, self.calib_params)
        phi_prime = err['phi_prime']; chi_prime = err['chi_prime']
        phi = err['phi']; chi = err['chi']
        ez = err['ez']; vz = err['vz']
        ey_prime = err['ey_prime']; ez_prime = err['ez_prime']
        vx_prime = err['vx_prime']; vz_prime = err['vz_prime']
        ey = err['ey']; vx = err['vx']
        A = err['_A']; sv = err['_sv']

        print("\n" + "="*40)
        print(f"  phi'     = {2*phi_prime:>+.5f} = {np.rad2deg(phi_prime):>+.5f} deg")
        print(f"  chi'     = {chi_prime:>+.5f} = {np.rad2deg(chi_prime):>+.5f} deg")
        print(f"  phi      = {phi:>+.5f}  = {np.rad2deg(phi):>+.5f} deg")
        print(f"  chi      = {chi:>+.5f}  = {np.rad2deg(chi):>+.5f} deg")
        print(f"  eps_z    = {ez:>+.5f} = {np.rad2deg(ez):>+.5f} deg")
        print(f"  vz       = {vz:>+.5f}  = {np.rad2deg(vz):>+.5f} deg")
        print(f"  eps_y'   = {ey_prime:>+.5f} = {np.rad2deg(ey_prime):>+.5f} deg")
        print(f"  eps_z'   = {ez_prime:>+.5f} = {np.rad2deg(ez_prime):>+.5f} deg")
        print(f"  vx'      = {vx_prime:>+.5f}  = {np.rad2deg(vx_prime):>+.5f} deg")
        print(f"  vz'      = {vz_prime:>+.5f} = {np.rad2deg(vz_prime):>+.5f} deg")
        print(f"  eps_y    = {ey:>+.5f} = {np.rad2deg(ey):>+.5f} deg")
        print(f"  vx       = {vx:>+.5f} = {np.rad2deg(vx):>+.5f} deg")
        print("="*40)
        print(f"  cond(A)  = {np.linalg.cond(A):.1f}")
        print(f"  sv       = {sv}")
        print(f"  max |error| = {max(abs(phi_prime), abs(chi_prime), abs(phi), abs(chi), abs(ez), abs(vz), abs(ey_prime), abs(ez_prime), abs(vx_prime), abs(vz_prime), abs(ey), abs(vx)):.5f} rad")

if __name__ == "__main__":
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    QUBIT = 'q1'
    SHOTS = 6000
    SIM_DURATION = 5_000 // 4  # clock cycles (4 ns each)

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {"threshold": 0.0, "ground_threshold": -0.1}

    # --- Run waveform simulation (no hardware needed) ---
    exp = BootstrapTomography(
        element=QUBIT,
        shots=SHOTS,
        calib_params=calib_params,
        sequence=DEFAULT_SEQUENCE,    # or pass a custom list of pulse tuples
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix='bootstrap_tomo_sim',
        config=config,
        simulate=False,
        sim_duration=SIM_DURATION,
    )

    # --- Run on hardware (uncomment to use) ---
    # exp = BootstrapTomography(
    #     element=QUBIT,
    #     shots=SHOTS,
    #     calib_params=calib_params,
    #     sequence=DEFAULT_SEQUENCE,
    #     meta_dict=meta_dict,
    #     outer_folder=meta_dict['global_params']['outer_folder'],
    #     path=QUBIT,
    #     suffix='bootstrap_tomo',
    #     config=config,
    # )
