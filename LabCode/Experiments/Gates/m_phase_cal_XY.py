from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg", force=True)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from LabCode.Helpers.pulse_macros import *
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class Phase_Calibration_XY(ExperimentClass):
    """
    Phase Calibration:
    Sweep phase_vec for multiple pulse sequences.
    Each pulse sequence is a tuple of operations.
    All sequences are played once per shot and plotted together.
    """
    def __init__(self, element, phase_vec, pulse_sequences, shots, calib_params,
                 outer_folder='', path='Phase_Cal', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)

        self.element = element
        self.phase_vec = phase_vec
        self.pulse_sequences = pulse_sequences  # List of tuples of pulse names
        self.shots = shots
        self.calib_params = calib_params
        self.read_len = self.meta_dict['global_params']['read_len']
        self.data = {
            'qubit': self.element,
            "shots": self.shots,
            'phase_vec': phase_vec,
            'pulse_sequences': pulse_sequences,
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
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]

        with program() as prog:
            n = declare(int)
            p = declare(fixed)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            n_stream = declare_stream()
            ss_dict={}
            for i in range(len(self.pulse_sequences)):
                ss_dict[f'{i}'] = declare_stream()

            with for_(n, 0, n < self.shots, n + 1):
                save(n, n_stream)
                with for_each_(p, self.phase_vec):
                    for i, sequence in enumerate(self.pulse_sequences):
                        reset_frame(qubit)
                        self.play_sequence(sequence, p, qubit)
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
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X90', qubit)
                                play('X90', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            wait(resonator_dump_time * u.ns, qubit)
                            assign(I_reset, I)

            with stream_processing():
                for i in range(len(self.pulse_sequences)):
                    ss_dict[f"{i}"].buffer(len(self.phase_vec)).average().save(f'ss_{i}')
                n_stream.save('n')
        return prog

    def analyze(self, job, qm):
        data_list = [f"ss_{i}" for i in range(len(self.pulse_sequences))] + ["n"]
        results = fetching_tool(job, data_list=data_list, mode="live")
        confusion_matrix = np.array(self.calib_params['confusion'])

        # Correct populations with confusion matrix
        c10 = confusion_matrix[1, 0]  # P(meas=1 | true=0)
        c11 = confusion_matrix[1, 1]  # P(meas=1 | true=1)
        # Live Plotting
        # --- create figure with stacked axes ---
        fig, ax1 = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"{self.element} Phase Calibration")

        # --- main population plot ---
        lines = [ax1.plot([], [], '.-', label=self.pulse_sequences[i])[0] for i in range(len(self.pulse_sequences))]

        ax1.set_ylabel("Population")
        ax1.set_xlabel("Phase (rad/2π)")
        ax1.set_ylim(0, 1)  # fix population y-axis
        ax1.legend(loc="upper right")

        # allow interrupt on figure close
        interrupt_on_close(fig, job)

        # --- live data fetching loop ---
        while results.is_processing():
            *populations, n = results.fetch_all()
            print(f"\rProgress: {int(n / self.shots * 100):3d}%", end="", flush=True)

            # --- process each population trace ---
            for i, raw_pop in enumerate(populations):
                # correct with confusion matrix
                population = np.clip((raw_pop - c10) / (c11 - c10), 0, 1)
                lines[i].set_data(self.phase_vec, population)
            self.data['ss_data'] = populations

            # --- refresh axes ---
            ax1.relim()
            ax1.autoscale_view(scaley=False)  # keep y fixed 0–1
            plt.pause(0.05)
            plt.pause(0.05)

        print("")  # flush console
        qm.close()
        fig.savefig(self.iname, bbox_inches="tight")


    def play_sequence(self, pulses, phase, qubit):
        """
        Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
        **Example:** I, Q = allXY(['I', 'y90'])

        :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
            operations in the config except for the identity operation that must be called 'I'.
        """
        for pulse in pulses:
            if pulse[0] == "Y":
                play("Y90", qubit)
                # play_pulse(qubit, pulse, phase)
            elif pulse[0] == "X":
                play("X90", qubit)
                # play_pulse(qubit, pulse, 0)
            else:
                wait(4, qubit)  # or wait if sequence is identity