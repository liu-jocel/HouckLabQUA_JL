from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
from LabCode.Helpers.pulse_macros import *
u = unit(coerce_to_integer=True)


class Frequency_Calibration_XY(ExperimentClass):
    """

    Frequency Calibtation:
    Choose a pulse sequence that compiles to Identity or 1 in Z.
    play: (pulse_sequence)**number_repeats vec.
    Should see scaling that can be reduced by changing Frequency
    """
    def __init__(self,element, f_vec, pulse_sequences, shots,  calib_params, num_repeats=1, outer_folder='', path='Phase_Cal', suffix='data', meta_dict=None, config=None, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config, **kwargs)

        self.element = element #meta dict element name
        self.f_vec = f_vec
        self.pulse_sequences = pulse_sequences # Array Like
        self.shots = shots
        self.num_repeats = num_repeats
        self.calib_params = calib_params
        self.read_len = self.meta_dict['global_params']['read_len']
        self.data = {
            'qubit': self.element,
            "shots": self.shots,
            'f_vec': f_vec,
            'pulse_sequence': pulse_sequences,
            "meta_dict": self.meta_dict
        }

        self.config = config
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        IF_freq = self.meta_dict['IF_freq']
        qb_drive_freq = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        with program() as prog:
            n = declare(int)
            df = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_dict={}
            for i in range(len(self.pulse_sequences)):
                ss_dict[f'{i}'] = declare_stream()
            n_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(df, self.f_vec):
                    for i,  sequence in enumerate(self.pulse_sequences):
                        update_frequency(qubit, df + IF_freq)
                        self.play_sequence(sequence, qubit, self.meta_dict)

                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_dict[f"{i}"])
                        align(qubit, resonator)

                        # Active reset
                        update_frequency(qubit, qb_drive_freq)
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)
                        #wait(3 * u.ms, qubit)

            with stream_processing():
                for i in range (len(self.pulse_sequences)):
                    ss_dict[f"{i}"].buffer(len(self.f_vec)).average().save(f'ss_{i}')
                n_stream.save('n')
        return prog

    def analyze(self, job, qm):
        data_list = [f"ss_{i}" for i in range(len(self.pulse_sequences))] + ["n"]
        results = fetching_tool(job, data_list=data_list, mode="live")
        confusion_matrix = np.array(self.calib_params['confusion'])

        # Correct populations with confusion matrix
        c10 = confusion_matrix[1, 0]  # P(meas=1 | true=0)
        c11 = confusion_matrix[1, 1]  # P(meas=1 | true=1)
        matplotlib.use("qtagg", force=True)

        # Live Plotting
        # --- create figure with stacked axes ---
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, figsize=(8, 6),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3}
        )
        fig.suptitle(f"{self.element} Frequency Calibration")

        # --- main population plot ---
        lines = [ax1.plot([], [], '.', label=self.pulse_sequences[i])[0] for i in range(len(self.pulse_sequences))]
        fit_lines = [ax1.plot([], [], '--', color='C' + str(i), alpha=0.7)[0] for i in range(len(self.pulse_sequences))]

        ax1.set_ylabel("Population")
        ax1.set_xlabel("Detuning (MHz)")
        ax1.set_ylim(-0.2, 1.2)  # fix population y-axis
        ax1.legend(loc="upper right")

        # --- bottom plot for min-max distance ---
        ax2.set_xlabel("Detuning (MHz)")
        ax2.set_ylabel("Max–Min Population")
        delta_line, = ax2.plot([], [], color="C0")
        ax2.grid(True, alpha=0.3)

        # allow interrupt on figure close
        interrupt_on_close(fig, job)

        # initialize variables
        populations = None
        optimal_detuning = self.meta_dict['detuning'] / u.MHz
        x_fine = np.linspace(self.f_vec[0] / u.MHz, self.f_vec[-1] / u.MHz, 5000)

        # create persistent vertical lines (update x-data in place each frame)
        vline1 = ax1.axvline(optimal_detuning, color="red", linestyle="--")
        vline2 = ax2.axvline(optimal_detuning, color="red", linestyle="--")

        # --- live data fetching loop ---
        while results.is_processing():
            *populations, n = results.fetch_all()
            print(f"\rProgress: {int(n / self.shots * 100):3d}%", end="", flush=True)

            regressions = [None] * len(populations)

            # --- process each population trace ---
            for i, raw_pop in enumerate(populations):
                # correct with confusion matrix
                population = (raw_pop - c10) / (c11 - c10)

                # linear regression
                regressions[i] = linregress(self.f_vec / u.MHz, population)
                y_fit = regressions[i].slope * x_fine + regressions[i].intercept

                # update lines
                lines[i].set_data(self.f_vec / u.MHz, population)
                fit_lines[i].set_data(x_fine, y_fit)

            # --- compute min-max distance ---
            x_grid = x_fine
            delta_y = np.full_like(x_grid, np.nan, dtype=float)
            for j, x in enumerate(x_grid):
                y_vals = [r.slope * x + r.intercept for r in regressions if r is not None]
                if len(y_vals) >= 2:
                    delta_y[j] = np.max(y_vals) - np.min(y_vals)

            # update delta plot
            delta_line.set_data(x_grid, delta_y)
            ax2.relim()
            ax2.autoscale_view()

            # --- find optimal IF ---
            idx = np.nanargmin(delta_y)
            x_opt = x_grid[idx]
            optimal_detuning = x_opt

            # --- update titles ---
            ax1.set_title(f"Optimal Detuning: {optimal_detuning:.3f} MHz")
            ax2.set_title("Population Spread vs Detuning")

            # update persistent vertical lines in place
            vline1.set_xdata([x_opt, x_opt])
            vline2.set_xdata([x_opt, x_opt])

            # store data
            self.data['ss_data'] = populations

            plt.pause(0.05)

        print("")  # flush console
        print(f"Optimal IF Frequency: {optimal_detuning} MHz")
        self.data['optimal_detuning_MHz'] = optimal_detuning
        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

    def play_sequence(self, pulses, qubit, meta_dict):
        """
        Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
        **Example:** I, Q = allXY(['I', 'y90'])

        :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
            operations in the config except for the identity operation that must be called 'I'.
        """
        for pulse in pulses:
            if pulse != "I":
                play_pulse(qubit, pulse, meta_dict[pulse]["phase"])  # Either play the sequence
            else:
                wait(4, qubit)  # or wait if sequence is identity