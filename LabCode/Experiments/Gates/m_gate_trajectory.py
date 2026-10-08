from qm.qua import *
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use("QtAgg", force= True)
from datetime import datetime
import matplotlib.pyplot as plt
plt.ion()
from qualang_tools.bakery import baking
from LabCode.Helpers.pulse_macros import *
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import convert_e_population_to_expectation
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
PHASE = 0
class GateTrajectory(ExperimentClass):
    """
    Amplified Phase Error Experiment implement from E. Lucero et. al:
    DOI: https://doi.org/10.1103/PhysRevA.82.042339
    """
    def __init__(self, element, pulse, n_times, shots, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.pulse = pulse
        self.t_vec = np.linspace(0, self.meta_dict[pulse]['length']/4, n_times, dtype=int)
        print(self.t_vec)
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            'shots': self.shots,
            'meta_dict': self.meta_dict,
            't_vec': self.t_vec
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
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
            X_stream = declare_stream()
            Y_stream = declare_stream()
            Z_stream = declare_stream()
            n_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)

                # Measure Z
                with for_each_(t, self.t_vec):
                    reset_frame(qubit)
                    # play_pulse(qubit, "X90", self.meta_dict[self.pulse]['phase'])
                    # play_pulse_duration(qubit, "Y90", t, self.meta_dict[self.pulse]['phase'])
                    play_pulse_duration(qubit, self.pulse, t, self.meta_dict[self.pulse]['phase'])

                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, Z_stream)
                    align(qubit, resonator)
                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play_pulse(qubit, "X180", self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
                    reset_frame()
                    # Measure Y

                    # play_pulse(qubit, "X90", self.meta_dict[self.pulse]['phase'])
                    # play_pulse_duration(qubit, "Y90", t, self.meta_dict[self.pulse]['phase'])
                    play_pulse_duration(qubit, self.pulse, t, self.meta_dict[self.pulse]['phase'])

                    play_pulse(qubit, "X90", self.meta_dict["X90"]['phase'])
                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, Y_stream)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play_pulse(qubit, "X180", self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)


                    # Measure X
                    reset_frame()

                    # play_pulse(qubit, "X90", self.meta_dict[self.pulse]['phase'])
                    # play_pulse_duration(qubit, "Y90", t, self.meta_dict[self.pulse]['phase'])
                    play_pulse_duration(qubit, self.pulse, t, self.meta_dict[self.pulse]['phase'])

                    play_pulse(qubit, "mY90", self.meta_dict["Y90"]['phase'])
                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    assign(I_reset, I)
                    with if_(I > ss_thresh):
                        assign(state, 1)
                    with else_():
                        assign(state, 0)
                    save(state, X_stream)
                    align(qubit, resonator)

                    # Active reset
                    with while_(I_reset > ground_confidence_threshold):
                        with if_(I_reset > ss_thresh):
                            play_pulse(qubit, "X180", self.meta_dict["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)


            with stream_processing():
                X_stream.buffer(len(self.t_vec)).average().save("X_ss")
                Y_stream.buffer(len(self.t_vec)).average().save("Y_ss")
                Z_stream.buffer(len(self.t_vec)).average().save("Z_ss")
                n_stream.save('n')
        return prog

    def analyze(self, job, qm):
        result = fetching_tool(self.job, data_list=["X_ss", "Y_ss", "Z_ss", "n"], mode="live")

        # --- Figure setup ---
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel(r"$\langle X \rangle$")
        ax.set_ylabel(r"$\langle Y \rangle$")
        ax.set_zlabel(r"$\langle Z \rangle$")
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.set_zlim([-1.05, 1.05])
        ax.set_title(f"Gate Trajectory on Bloch Sphere. {self.pulse}")

        # --- Bloch sphere ---
        u_s = np.linspace(0, 2 * np.pi, 60)
        v_s = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u_s), np.sin(v_s))
        ys = np.outer(np.sin(u_s), np.sin(v_s))
        zs = np.outer(np.ones_like(u_s), np.cos(v_s))
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.25, linewidth=0.6)

        # Axes
        ax.plot([-1, 1], [0, 0], [0, 0], "k--", lw=1, alpha=0.5)
        ax.plot([0, 0], [-1, 1], [0, 0], "k--", lw=1, alpha=0.5)
        ax.plot([0, 0], [0, 0], [-1, 1], "k--", lw=1, alpha=0.5)

        # --- Colormap for time ---
        cmap = plt.cm.viridis
        norm = plt.Normalize(0, len(self.t_vec))

        (traj_line,) = ax.plot([], [], [], "-", lw=1.5, color="gray", alpha=0.7)
        scat = ax.scatter([], [], [], c=[], cmap=cmap, norm=norm, s=30)

        cbar = plt.colorbar(scat, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Time (clock cycles)")

        plt.tight_layout()
        plt.show(block=False)

        # ----------- CONTROL FLAGS -----------
        keep_plot_open = True

        def on_close(event):
            nonlocal keep_plot_open
            keep_plot_open = False

        fig.canvas.mpl_connect("close_event", on_close)

        # confusion matrix for ccorrecting data
        confusion_matrix = self.calib_params['confusion']
        true_positive = confusion_matrix[1][1]
        false_positive = confusion_matrix[0][1]

        # ----------- LIVE LOOP -----------
        while keep_plot_open:

            if result.is_processing():
                pop_X, pop_Y, pop_Z, n = result.fetch_all()

                # # correct with confusion matrix
                # pop_X_corrected = np.clip((pop_X - false_positive) / (true_positive - false_positive), 0, 1)
                # pop_Y_corrected = np.clip((pop_Y - false_positive) / (true_positive - false_positive), 0, 1)
                # pop_Z_corrected = np.clip((pop_Z - false_positive) / (true_positive - false_positive), 0, 1)

                expect_X = convert_e_population_to_expectation(pop_X)
                expect_Y = convert_e_population_to_expectation(pop_Y)
                expect_Z = convert_e_population_to_expectation(pop_Z)

                t_clk = np.arange(len(expect_X))

                traj_line.set_data(expect_X, expect_Y)
                traj_line.set_3d_properties(expect_Z)
                scat._offsets3d = (expect_X, expect_Y, expect_Z)
                scat.set_array(t_clk)

                self.data.update({
                    "X": pop_X,
                    "Y": pop_Y,
                    "Z": pop_Z,
                    "expect_X": expect_X,
                    "expect_Y": expect_Y,
                    "expect_Z": expect_Z,
                })

            plt.pause(0.05)
            plt.pause(0.05)

        # ----------- CLEAN EXIT -----------
        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)


