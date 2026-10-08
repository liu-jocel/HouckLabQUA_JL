from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
# All XY sequences. The sequence names must match corresponding operation in the config
sequence = [
    ("I", "I"),
    ("X90", "X90", "X90", "X90"),
    ("Y90", "Y90", "Y90", "Y90"),
    ("X90", "X90", "Y90", "Y90"),
    ("Y90", "Y90", "X90", "X90"),
    ("X90", "I"),
    ("Y90", "I"),
    ("X90", "Y90"),
    ("Y90", "X90"),
    ("X90", "Y90", "Y90"),
    ("Y90", "X90", "X90"),
    ("X90", "X90", "Y90"),
    ("Y90", "Y90", "X90"),
    ("X90", "X90", "X90"),
    ("Y90", "Y90", "Y90"),
    ("X90", "X90", "I"),
    ("Y90", "Y90", "I"),
    ("X90", "X90"),
    ("Y90", "Y90"),
]
# sequence = [    ("X90", "X90"),
#                 ("Y90", "Y90"),
#                 ("Y90", "Y90", "X90", "X90"),
#                 ("X90", "Y90"),
#                 ("X90", "Ym90"),
#                 ("Xm90", "Y90"),
#                 ("Xm90",  "Ym90"),
#                 ("Y90", "X90"),
#                 ("Y90", "Xm90"),
#                 ("Ym90", "X90"),
#                 ("Ym90", "Xm90"),
#                 ("X90", "I"),
#                 ("Xm90", "I"),
#                 ("Y90", "I"),
#                 ("Ym90", "I"),
#                 ("Xm90", "Y90", "X90"),
#                 ("Xm90", "Ym90", "X90"),
#                 ("X90", "X90", "Y90"),
#                 ("X90", "X90", "Ym90"),
#                 ("Y90", "Y90", "X90"),
#                 ("Y90", "Y90", "Xm90"),
#                 ("X90", "Y90", "X90"),
#                 ("Xm90", "Y90", "Xm90")]


class AllXYDetuning(ExperimentClass):
    """
    All XY Experiment
    """

    def __init__(self, element, f_vec, shots, calib_params, num_repeats=1, outer_folder='', path='', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element
        self.f_vec = f_vec
        self.meta_dic = meta_dict
        self.config = config
        self.shots = shots
        self.num_repeats = num_repeats
        self.calib_params = calib_params

        self.data = {
            "meta_dict": meta_dict,
            'f_vec': f_vec,
            'config': config,
            "shots": shots,
            "num_pulses": num_repeats,
            'calib_params': calib_params
        }
        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

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
        IF_freq = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        shots = self.shots
        num_repeats = self.num_repeats

        with program() as prog:
            n = declare(int)
            i = declare(int)
            m = declare(int)
            df = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            n_stream = declare_stream()
            ss_dict = {}
            for i, f in enumerate(self.f_vec):
                ss_dict[f'{i}'] = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                assign(i, 0)
                with for_each_(df, self.f_vec):
                    for gates in sequence:
                        align()
                        update_frequency(qubit, df + IF_freq)
                        with for_(m, 0, m < num_repeats, n + 1):
                            self.allXY(gates, qubit)

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

                        update_frequency(qubit, IF_freq)
                        # Active reset
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)
                    assign(i, i+1)
            with stream_processing():
                for i in range(len(self.f_vec)):
                    ss_dict[f"{i}"].buffer(len(sequence)).average().save(f'ss_{i}')
        return prog

    def analyze(self, job, qm):
        data_list = [f"ss_{f}" for f in range(len(self.f_vec))] + ["n"]
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
        lines = [ax1.plot([], [], '.', label=self.f_vec[i])[0] for i in range(len(self.f_vec))]

        ax1.set_ylabel("Population")
        ax1.set_xlabel("Detuning (MHz)")
        ax1.set_ylim(0, 1)  # fix population y-axis
        ax1.legend(loc="upper right")

        # allow interrupt on figure close
        interrupt_on_close(fig, job)
        # Fetch data, store and update plot.
        while results.is_processing():
            populations_raw = results.fetch_all()
            p1_meas = populations_raw  # array
            populations = (p1_meas - c10) / (c11 - c10)
            populations = np.clip(populations, 0.0, 1.0)

        qm.close()
        plt.show()
        plt.savefig(self.iname, bbox_inches="tight")
        # plt.ioff()
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
            if pulse != "I":
                play(pulse, qubit)  # Either play the sequence
            else:
                wait(4, qubit)  # or wait if sequence is identity