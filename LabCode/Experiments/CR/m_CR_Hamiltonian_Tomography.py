from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import functools
import time
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.CR_Helpers import *
from LabCode.Helpers.cr_hamiltonian_tomography import (
    CRHamiltonianTomographyAnalysis,
    plot_crqst_result_2D,
    plot_crqst_result_3D,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.results.data_handler import DataHandler
from qualang_tools.loops import from_array
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


class HamiltonianTomographyCR(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """

    def __init__(self, t_vec, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
                 suffix='data', cancel=False, echo=False):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix, meta_dict=meta_dict, config=config)
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

        self.t_vec = t_vec
        self.cancel = cancel
        self.echo = echo

        if self.cancel and self.echo:
            self.job = self.acquire(self.make_prog_echo_cancel(), self.config)
            self.t_vec = 2 * t_vec
        elif self.cancel:
            self.job = self.acquire(self.make_prog_cancel(), self.config)
        elif self.echo:
            self.job = self.acquire(self.make_prog_echo(), self.config)
            self.t_vec = 2 * t_vec
        else:
            self.job = self.acquire(self.make_prog(), self.config)

        self.data = {
            "shots": self.shots,
            "t_vec": self.t_vec,
            "meta_dict": self.meta_dict,
            'calib_params': self.calib_params,
            'confusion': self.confusion,
            'cancel': cancel,
            'echo': echo
        }

        self.analyze()
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
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
            t = declare(int)
            control_state = declare(int)  # 0 or 1, State of control qubit
            proj = declare(int)  # 0, 1, 2, For projection onto X, Y, Z axis
            state_qt = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)
            state_stream_qt = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # CR
                            play("ZX90", qc, duration=t)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    # play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    # play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
                            align(q1, q2, qc, qt, r1, r2)
                            I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                            assign(I_reset_1, I1)
                            assign(I_reset_2, I2)

                            # State Discrimination
                            with if_(I2 > ss_thresh_q2):
                                assign(state_qt, 1)
                            with else_():
                                assign(state_qt, 0)
                            save(state_qt, state_stream_qt)
                            align(q1, q2, qc, qt, r1, r2)

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_qt.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
        return prog

    def make_prog_cancel(self):
        """
        Qua program for CR experiment with compensation pulse on qt
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
            t = declare(int)
            control_state = declare(int)  # 0 or 1, State of control qubit
            proj = declare(int)  # 0, 1, 2, For projection onto X, Y, Z axis
            state_qt = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)
            state_stream_qt = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Cancel CR
                            play("ZX90", qc, duration=t)
                            play("ZX90", qt, duration=t)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    # play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    # play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
                            align(q1, q2, qc, qt, r1, r2)
                            I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                            assign(I_reset_1, I1)
                            assign(I_reset_2, I2)

                            # State Discrimination
                            with if_(I2 > ss_thresh_q2):
                                assign(state_qt, 1)
                            with else_():
                                assign(state_qt, 0)
                            save(state_qt, state_stream_qt)
                            align(q1, q2, qc, qt, r1, r2)

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_qt.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
        return prog

    def make_prog_echo(self):
        """
        Qua program for Echo CR experiment
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
            t = declare(int)
            control_state = declare(int)  # 0 or 1, State of control qubit
            proj = declare(int)  # 0, 1, 2, For projection onto X, Y, Z axis
            state_qt = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)
            state_stream_qt = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Echo CR
                            play("ZX45", qc, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)
                            play("ZXm45", qc, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    # play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    # play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
                            align(q1, q2, qc, qt, r1, r2)
                            I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                            assign(I_reset_1, I1)
                            assign(I_reset_2, I2)

                            # State Discrimination
                            with if_(I2 > ss_thresh_q2):
                                assign(state_qt, 1)
                            with else_():
                                assign(state_qt, 0)
                            save(state_qt, state_stream_qt)
                            align(q1, q2, qc, qt, r1, r2)

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_qt.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
        return prog

    def make_prog_echo_cancel(self):
        """
        Qua program for CR experiment with echo and cancellation tone on qt
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
            t = declare(int)
            control_state = declare(int)  # 0 or 1, State of control qubit
            proj = declare(int)  # 0, 1, 2, For projection onto X, Y, Z axis
            state_qt = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_reset_1 = declare(fixed)
            I_reset_2 = declare(fixed)
            state_stream_qt = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(t, self.t_vec * u.ns)):
                    with for_(proj, 0, proj < 3, proj + 1):  # bases
                        with for_(control_state, 0, control_state < 2, control_state + 1):  # states
                            # State Preparation
                            with if_(control_state == 1):
                                play("X180", q1)
                            align(q1, q2, qc, qt)

                            # CR Echo
                            play("ZX45", qc, duration=t)
                            play("ZX45", qt, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)
                            play("ZXm45", qc, duration=t)
                            play("ZXm45", qt, duration=t)
                            align(q1, q2, qc, qt)
                            play("X180", q1)
                            align(q1, q2, qc, qt)

                            # Projective Measurements
                            with switch_(proj):
                                with case_(0):  # projection along X
                                    # play("Ym90", q1)
                                    play("Ym90", q2)
                                with case_(1):  # projection along Y
                                    # play("X90", q1)
                                    play("X90", q2)
                                with case_(2):  # projection along Z
                                    wait(4, q2)
                            align(q1, q2, qc, qt, r1, r2)
                            I1, Q1, I2, Q2 = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)

                            assign(I_reset_1, I1)
                            assign(I_reset_2, I2)

                            # State Discrimination
                            with if_(I2 > ss_thresh_q2):
                                assign(state_qt, 1)
                            with else_():
                                assign(state_qt, 0)
                            save(state_qt, state_stream_qt)

                            align(q1, q2, qc, qt, r1, r2)

                            # Active reset
                            with while_((I_reset_1 > ground_confidence_threshold_q1)
                                        | (I_reset_2 > ground_confidence_threshold_q2)):
                                with if_(I_reset_1 > ss_thresh_q1):
                                    play('X180', q1, )
                                with if_(I_reset_2 > ss_thresh_q2):
                                    play('X180', q2, )
                                align(q1, q2, qc, qt, r1, r2)
                                I1, _, I2, _ = measure_2q(element_q1, element_q2, I1, Q1, I2, Q2)
                                assign(I_reset_1, I1)
                                assign(I_reset_2, I2)
                            wait(resonator_dump_time * u.ns, q1)

            with stream_processing():
                state_stream_qt.buffer(2).buffer(3).buffer(len(self.t_vec)).average().save("ss1")
        return prog

    def analyze(self):
        """
        Function to analyze results of Hamiltonian Tomography Experiment.
        """
        results = fetching_tool(self.job, data_list=["ss1"], mode='live')
        try:
            while results.is_processing():
                # Fetch results
                res = results.fetch_all()
                state_t = res[0]
                # Progress bar
                # progress_counter(iterations, self.shots, start_time=results.start_time)
                bloch_t = -2 * state_t + 1  # convert |0> -> 1, |1> -> -1
                # Progress bar
                # progress_counter(iterations, elf.shots, start_time=results.start_time)
                # plotting data
                # cross resonance Hamiltonian tomography analysis
                crht = CRHamiltonianTomographyAnalysis(
                    ts=self.t_vec,
                    data=bloch_t,  # target data
                )
                crht.fit_params()
                fig_analysis = crht.plot_fit_result(do_show=False)
                fig_analysis.savefig(self.iname[:-4])
                self.data["ham_tomo_params_fitted"] = crht.params_fitted_dict
                self.data["ham_tomo_interaction_coeffs_MHz"] = crht.interaction_coeffs_MHz

                # Save results
                self.h5_data()
        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            res = results.fetch_all()
            state_t = res[0]
            bloch_t = -2 * state_t + 1
            crht = CRHamiltonianTomographyAnalysis(
                ts=self.t_vec,
                data=bloch_t,  # target data
            )
            crht.fit_params()
            fig_analysis = crht.plot_fit_result()
            # self.data["fig_live"] = fig
            fig_analysis.savefig(self.iname[:-4])
            self.data["fig_analysis"] = fig_analysis
            self.data["ham_tomo_params_fitted"] = crht.params_fitted_dict
            self.data["ham_tomo_interaction_coeffs_MHz"] = crht.interaction_coeffs_MHz

            # Save results
            self.h5_data()
            print("Experiment QM is now closed")
