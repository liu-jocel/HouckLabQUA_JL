from qm.qua import *
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.SPAM_correction import correct_counts_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class CNOT(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """
    def __init__(self, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
                 suffix='data', cancel=False, echo=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion
        self.read_len = self.meta_dict['global_params']['read_len']

        self.q1 = self.meta_dict['q1']['q_name']
        self.q2 = self.meta_dict['q2']['q_name']
        self.qc = self.meta_dict['qc']['q_name']
        self.qt = self.meta_dict['qt']['q_name']
        self.r1 = self.meta_dict['q1']['r_name']
        self.r2 = self.meta_dict['q2']['r_name']

        self.data = {
            "shots": self.shots,
            "meta_dict": self.meta_dict,
            'calib_params': self.calib_params,
            'confusion': self.confusion,
            'cancel': cancel,
            'echo': echo
        }
        self.cancel = cancel
        self.echo = echo
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        """
        Qua Program for Standard CR experimenet
        """
        element_q1 = self.meta_dict['q1']
        element_q2 = self.meta_dict['q2']
        q1 = self.q1
        q2 = self.q2
        qc = self.qc
        qt = self.qt
        r1 = self.r1
        r2 = self.r2
        resonator_dump_time = self.meta_dict[q1]['resonator_dump_time']
        shots = self.shots
        ss_thresh_q1 = self.calib_params["threshold_q1"]
        ss_thresh_q2 = self.calib_params["threshold_q2"]
        ground_confidence_threshold_q1 = self.calib_params["ground_threshold_q1"]
        ground_confidence_threshold_q2 = self.calib_params["ground_threshold_q2"]

        with program() as prog:
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            n = declare(int)
            n_st = declare_stream()
            state_1 = declare(int)
            state_2 = declare(int)
            state_stream_1 = declare_stream()
            state_stream_2 = declare_stream()
            c = declare(int)
            with_cnot = declare(int)


            with for_(n, 0, n < shots, n + 1):
                save(n, n_st)
                # to allow time to save the data
                wait(1000 * u.ns)
                with for_(c, 0, c < 4, c + 1):
                    with for_(with_cnot, 0, with_cnot < 2, with_cnot + 1):
                        # QST on Target
                        with switch_(c):
                            with case_(0):  # projection along X
                                wait(4, q1)
                            with case_(1):  # projection along Y
                                play("X180", q2)
                                align()
                            with case_(2):  # projection along Z
                                play("X180", q1)
                                align()
                            with case_(3):  # projection along Z
                                play("X180", q1)
                                play("X180", q1)
                                align()

                        with if_(with_cnot == 1):
                            frame_rotation_2pi(+0.25, 'q1')  # +0.25 for Z(-pi/2)
                            play("Xm90", 'q2')

                            # Shift frames to the calibrated phases
                            # Play CR
                            align("q1", "q2", "qc", "qt", "r1", "r2")
                            # CR Echo
                            play("ZX45", "qc", )
                            play("ZX45", "qt", )
                            align("q1", "q2", "qc", "qt")
                            play("X180", "q1")
                            align("q1", "q2", "qc", "qt")
                            play("ZXm45", "qc", )
                            play("ZXm45", "qt", )
                            align("q1", "q2", "qc", "qt")
                            play("X180", "q1")
                            align("q1", "q2", "qc", "qt")

                            # Shift frames to the calibrated phases
                            # frame_rotation_2pi(cr_drive_phase_ZI_correct, 'qc')
                            align("q1", "q2", "qc", "qt", "r1", "r2")
                            frame_rotation_2pi(+0.25, 'q1')
                            # check the sign before 0.25 -- could be either way depending on 'qc'_phase
                            play("Y90" * amp(0.00001), 'q1', duration=1)
                            # this is a tricky step. the goal is to avoid face-to-face frame rotations
                            # qm only recognizes the last frame rotation before a real pulse if multiple frame rotations are stacked
                            # e.g., two CNOTs can be adjacent in a random gate sequence, and two 0.25 rotations are intended
                            # but qm only implements the last 0.25 rotation

                            # if you want to avoid this subtlety, you can use real XY gates to replace the virtual Z/2
                            # Z/2: play('Xm90', 'q1'); play('Y90', 'q1'); play('X90', 'q1');
                            # -Z/2: play('Xm90', 'q1'); play('Ym90', 'q1'); play('X90', 'q1');

                            align("q1", "q2", "qc", "qt", "r1", "r2")

                        I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                        # State Discrimination
                        assign(state_1, I1 > ss_thresh_q1)
                        assign(state_2, I2 > ss_thresh_q2)
                        save(state_1, state_stream_1)
                        save(state_2, state_stream_2)

                        align(q1, q2, qc, qt, r1, r2)

                        # Active reset
                        with while_((I1 > ground_confidence_threshold_q1)
                                    | (I2 > ground_confidence_threshold_q2)):
                            with if_(I1 > ss_thresh_q1):
                                play('X180', q1, )
                            with if_(I1 > ss_thresh_q2):
                                play('X180', q2, )
                            align(q1, q2, qc, qt, r1, r2)
                            I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                        wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                n_st.save("iteration")
                # control qubit
                state_stream_1.boolean_to_int().buffer(2).buffer(4).save_all("state1")
                # target qubit
                state_stream_2.boolean_to_int().buffer(2).buffer(4).save_all("state2")
            return prog

    def analyze(self):
        fig = plt.figure()
        # Tool to easily fetch results from the OPX (results_handle used in it)
        fetch_names = ["iteration", "state1",  "state2"]
        results = fetching_tool(self.job, fetch_names, mode="live")
        # Live plotting
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            iterations, I1, Q1, state_c, I2, Q2, state_t = res
            # Progress bar
            progress_counter(iterations, self.shots, start_time=results.start_time)
            self.data["ss_raw"] = {
                "qc": fetch_names[1],
                "qt": fetch_names[2]
            }
        data = []
        for i in range(2):
            cnot = {0: "N", 1: "Y"}[i]
            for j in range(4):
                case = {0: "00", 1: "01", 2: "10", 3: "11"}[j]
                for k in range(self.shots):
                    c, t = state_c[k, j, i], state_t[k, j, i]
                    data.append([cnot, case, f"{c}{t}"])
        df = pd.DataFrame(data, columns=["cnot", "case", "res"])

        # Plot bar plot for side-by-side comparison
        fig_analysis, axss = plt.subplots(2, 2, figsize=(10, 8))
        cases = ["00", "01", "10", "11"]
        for ax, case in zip(axss.ravel(), cases):
            case_data = df[df["case"] == case]  # Filter data for the current "case"

            # Group by "cnot" and "res" and count occurrences
            grouped = case_data.groupby(["cnot", "res"]).size().unstack(fill_value=0)

            # Generate side-by-side bar plot
            width = 0.35  # Width of bars
            index = np.arange(len(grouped.columns))  # The x locations for the groups

            # Bar plot for cnot=True
            ax.bar(index - width / 2, grouped.loc["N"], width, label="cnot=No")

            # Bar plot for cnot=False
            ax.bar(index + width / 2, grouped.loc["Y"], width, label="cnot=Yes")

            # Add some text for labels, title, and axes ticks
            ax.set_xlabel("res")
            ax.set_ylabel("Count")
            ax.set_title(f"Counts of states in case of preparing {case}")
            ax.set_xticks(index)
            ax.set_xticklabels(grouped.columns)
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.iname[:-4], bbox_inches='tight')
        plt.close(fig)
        self.data.update({
            "counts": df.to_dict()})
        self.h5_data()