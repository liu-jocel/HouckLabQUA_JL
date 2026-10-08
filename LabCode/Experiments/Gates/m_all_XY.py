from qm.qua import *
import numpy as np
import h5py
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from LabCode.Helpers.pulse_macros import *
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
# All XY sequences. The sequence names must match corresponding operation in the config
# sequence = [

sequence = [
    ("I", "I"),
    ("X90", "X90", "X90", "X90",),
    ("Y90", "Y90", "Y90", "Y90",),
    ("X90", "X90", "Y90", "Y90",),
    ("Y90", "Y90", "X90", "X90",),
    ("X90", "I"),
    ("Y90", "I"),
    ("X90", "Y90"),
    ("Y90", "X90"),
    ("X90", "Y90", "Y90",),
    ("Y90", "X90", "X90",),
    ("X90", "X90", "Y90"),
    ("Y90", "Y90", "X90"),
    ("X90", "X90", "X90"),
    ("Y90", "Y90", "Y90"),
    ("X90", "X90", "I"),
    ("Y90", "Y90", "I"),
    ("X90", "X90",),
    ("Y90", "Y90"),]

class AllXY(ExperimentClass):
    """
    All XY Experiment
    """

    def __init__(self, element, shots, calib_params, num_pulses=1, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.meta_dic = meta_dict
        self.config = config
        self.shots = shots
        self.num_pulses = num_pulses
        self.calib_params = calib_params

        self.data = {
            "meta_dict": meta_dict,
            'config': config,
            "shots": shots,
            "num_pulses": num_pulses,
            'calib_params': calib_params
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()
        #super().h5_data()

    def make_prog(self):
        """
        Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.

        :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
            operations in the config except for the identity operation that must be called 'I'.
        """
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        shots = self.shots
        num_pulses = self.num_pulses

        with program() as prog:
            n = declare(int)
            m = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            ss_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                for i in range(len(sequence)):
                    reset_frame(qubit)
                    align()
                    with for_(m, 0, m < num_pulses, n + 1):
                        self.allXY(sequence[i], qubit)
                    # SS Readout
                    align(qubit, resonator)
                    I, Q = measure_1q(self.meta_dict, I, Q)
                    save(I, I_stream)
                    save(Q, Q_stream)
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
                            play_pulse(qubit, "X180", self.meta_dic["X180"]['phase'])
                            align(qubit, resonator)
                        I, _ = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                    wait(resonator_dump_time * u.ns, qubit)
                    # wait(3 * u.ms, qubit)
            with stream_processing():
                ss_stream.buffer(len(sequence)).average().save("populations")
                I_stream.buffer(len(sequence)).save_all("I")
                Q_stream.buffer(len(sequence)).save_all("Q")
        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=['populations', "I", "Q",], mode="live")
        x_labels = [" ".join(pulses) for pulses in sequence]
        x = range(len(sequence))
        self.data['sequence'] = x_labels

        matplotlib.use("qtagg", force=True)
        # Live Plotting
        fig, ax = plt.subplots()
        hline = ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="0.5 Reference")
        (line,) = ax.plot(x, [0.5]*len(sequence), "o-", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Sequence")
        ax.set_ylabel("Population")
        ax.set_ylim(-0.5, 1.5)  # typical population range
        ax.set_title(f"All XY Sequences, {self.num_pulses} repetitions")
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

        # Fetch data, store and update plot.
        while results.is_processing():
            populations_raw, I, Q = results.fetch_all()
            self.data['I'] = I
            self.data['Q'] = Q
            confusion_matrix = np.array(self.calib_params['confusion'])

            # Correct populaitons with confusion matrix
            c10 = confusion_matrix[1, 0]  # P(meas=1 | true=0)
            c11 = confusion_matrix[1, 1]  # P(meas=1 | true=1)
            p1_meas = populations_raw  # array
            populations = (p1_meas - c10) / (c11 - c10)
            # populations = np.clip(populations, 0.0, 1.0) #is this necessary or artificially hiding a possible error?JL
            self.data['populations'] = populations

            line.set_data(x, populations)
            plt.pause(0.05)
        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        self.data['populations'] = populations_raw

    def allXY(self, pulses, qubit):
        """
        Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
        **Example:** I, Q = allXY(['I', 'y90'])

        :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
            operations in the config except for the identity operation that must be called 'I'.
        """
        # play("X90", qubit)
        for pulse in pulses:
            if pulse.upper() == "I":
                wait(4, qubit)  # or wait if sequence is identity
            else:
                play_pulse(qubit, pulse, self.meta_dic[pulse]['phase'])
