from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import functools
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.CR_Helpers import *
from qualang_tools.results import progress_counter, fetching_tool
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RamseyCR(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """
    def __init__(self, phase_vec, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
                 suffix='data', cancel=False, echo=False):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion

        self.q1 = self.meta_dict['q1']['q_name']
        self.q2 = self.meta_dict['q2']['q_name']
        self.qc = self.meta_dict['qc']['q_name']
        self.qt = self.meta_dict['qt']['q_name']
        self.r1 = self.meta_dict['q1']['r_name']
        self.r2 = self.meta_dict['q2']['r_name']

        self.phase_vec = phase_vec

        self.data = {
            "shots": self.shots,
            "t_vec": self.phase_vec,
            "meta_dict": self.meta_dict,
            'calib_params': self.calib_params,
            'confusion': self.confusion,
            'cancel': cancel,
            'echo': echo
        }
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
            n = declare(int)
            s = declare(int)  # 0:s, 1:e for control state
            ph = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            state_qc = declare(int)
            state_qt = declare(int)
            n_st = declare_stream()
            qc_stream = declare_stream()
            qt_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                # Save the averaging iteration to get the progress bar
                save(n, n_st)
                with for_(*from_array(ph, self.phase_vec)):
                    with for_(s, 0, s < 2, s + 1):  # states
                        # Prepare Qt to |1>
                        with if_(s == 1):
                            play("X180", q2)
                        with else_():
                            align(q1, q2, qc, qt, r1, r2)

                        # Bring Qc to x axis
                        play("Y90", q1)

                        align()
                        # direct + cancel
                        align(q1, q2, qc, qt, r1, r2)
                        play("ZX45", qc)
                        play("ZX45", qt)
                        # pi pulse on control
                        align(q1, q2, qc, qt, r1, r2)
                        play("X180", q1)
                        # echoed direct + cancel
                        align(q1, q2, qc, qt, r1, r2)
                        play("ZXm45", qc)
                        play("ZXm45", qt)
                        # pi pulse on control
                        align(q1, q2, qc, qt, r1, r2)
                        play("X180", q1)
                        align()
                        reset_frame(qc)
                        reset_frame(qt)

                        # phase shift
                        frame_rotation(ph, qc)

                        # Bring Qc back to z axis
                        play("Ym90", q1)

                        # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                        align(q1, q2, qc, qt, r1, r2)
                        I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                        #
                        # State Discrimination
                        with if_(I1 > ss_thresh_q1):
                            assign(state_qc, 1)
                        with else_():
                            assign(state_qc, 0)
                        save(state_qc, qc_stream)
                        with if_(I2 > ss_thresh_q2):
                            assign(state_qt, 1)
                        with else_():
                            assign(state_qt, 0)
                        save(state_qt, qc_stream)

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
                n_st.save("n")
                # control qubit
                qc_stream.buffer(2).buffer(len(ts_cycles)).buffer(len(self.phase_vec)).average().save(
                    "state1")
                # target qubit
                qt_stream.buffer(2).buffer(len(ts_cycles)).buffer(len(self.phase_vec)).average().save(
                    "state2")

        return prog

    def analyze(self):
        """
        Function to analyze results of Hamiltonian Tomography Experiment.
        """
        fetch_names = ["n", "qc_stream", "qt_stream"]
        results = fetching_tool(self.job, fetch_names, mode="live")
        # Prepare the figure for live plotting
        fig, axss = plt.subplots(2, 3, figsize=(8, 6))
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            iterations, state_c, state_t = res
            # Progress bar
            progress_counter(iterations, self.shots, start_time=results.start_time)
            # Progress bar
            progress_counter(iterations, self.shots, start_time=results.start_time)
            # Live plot data
            fig.suptitle(f"Phase calibration for qc")
            for ax, fname, V in zip(axss.ravel(), fetch_names[1:], res[1:]):
                ax.plot(self.phase_vec, V[:, 0])
                ax.plot(self.phase_vec, V[:, 1])
                ax.set_xlabel("Phase [2pi rad.]")
                ax.set_ylabel(fname.replace("1", "c").replace("2", "t"))
                ax.set_title(fname)
                ax.legend(["qc=|0>", "qc=|1>"])
            fig.tight_layout()
            plt.pause(1)
            self.data["qc_population"] = state_c
            self.data["qt_population"] = state_t

            plt.tight_layout()
            plt.savefig(self.iname[:-4], bbox_inches='tight')
            plt.close(fig)
            self.h5_data()