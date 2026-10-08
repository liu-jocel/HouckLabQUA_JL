from qm.qua import *
import numpy as np
import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import functools
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_2q
from LabCode.Helpers.CR_Helpers import *
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class HEATCR(ExperimentClass):
    """
    Rabi chevron experiemt with single shot readout
    """
    def __init__(self, t_vec, num_pulses, shots, calib_params, confusion, meta_dict, config, outer_folder='', path='CR',
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
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        # super().pickle_data()

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
                            play("ZX45", qc, duration=t)
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
                            play("ZX45", qc, duration=t)
                            play("ZX45", qt, duration=t)
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
        result = fetching_tool(self.job, data_list=["ss1"])
        excited_probs_qt = result.fetch_all()[0]
        # SS data for each state preparation and measurement projection

        meas_results = {
            '<X>t': {
                "qc = 0": -2 * excited_probs_qt[:, 0, 0, ] + 1,
                "qc = 1": -2 * excited_probs_qt[:, 0, 1, ] + 1,
            },
            '<Y>t': {
                "qc = 0": -2 * excited_probs_qt[:, 1, 0, ] + 1,
                "qc = 1": -2 * excited_probs_qt[:, 1, 1, ] + 1,
            },
            '<Z>t': {
                "qc = 0": -2 * excited_probs_qt[:, 2, 0, ] + 1,
                "qc = 1": -2 * excited_probs_qt[:, 2, 1, ] + 1,
            },
        }

        # Calculate two-norm of result
        R = np.sqrt((meas_results["<X>t"]["qc = 0"] + meas_results["<X>t"]["qc = 1"]) ** 2 +
                    (meas_results["<Y>t"]["qc = 0"] + meas_results["<Y>t"]["qc = 1"]) ** 2 +
                    (meas_results["<Z>t"]["qc = 0"] + meas_results["<Z>t"]["qc = 1"]) ** 2)
        R = R / 2
        try:
            CR_len_idx = find_first_minima_idx(R, prominence=0.2)
        except:
            CR_len_idx = find_first_minima_idx(R, prominence=0)

        self.data.update({"CR_len": self.t_vec[CR_len_idx]})

        popt_ground, pcov_ground, popt_excited, pcov_excited = self.fit_hamiltonian_tomography(meas_results)
        interaction_rates = calculate_interaction_rates(popt_ground, pcov_ground, popt_excited, pcov_ground)

        # Adjust t_vec to compensate for echo in plotting
        meas_results['<X>t']['fit_ground'] = Fx(self.t_vec, *popt_ground)
        meas_results['<X>t']['fit_excited'] = Fx(self.t_vec, *popt_excited)
        meas_results['<Y>t']['fit_ground'] = Fy(self.t_vec, *popt_ground)
        meas_results['<Y>t']['fit_excited'] = Fy(self.t_vec, *popt_excited)
        meas_results['<Z>t']['fit_ground'] = Fz(self.t_vec, *popt_ground)
        meas_results['<Z>t']['fit_excited'] = Fz(self.t_vec, *popt_excited)

        # Create a 1, 4 grid of subplots
        fig, axs = plt.subplots(4,1, figsize=(10, 12))
        # Flatten the axes array for easy iteration
        axs = axs.flatten()
        t_vec = self.t_vec / u.ns
        for i, (title, data) in enumerate(meas_results.items()):
            # Iterate over the dictionary and plot each key-value pair
            axs[i].plot(t_vec, data["qc = 0"], "o", color="blue", label="qc = 0")
            axs[i].plot(t_vec, data["qc = 1"], "o", color="red", label="qc = 1")
            axs[i].plot(t_vec, data["fit_ground"], "-", color="blue", label="fit qc = 0")
            axs[i].plot(t_vec, data["fit_excited"], "-", color="red", label="fit qc = 1")
            axs[i].axvline(x=t_vec[CR_len_idx], color='black', linestyle='--')
            axs[i].legend()
            axs[i].set_ylim(-1, 1)
            axs[i].set_ylabel(title)
        axs[3].plot(t_vec, R, "o", color="black")
        axs[3].axvline(x=t_vec[CR_len_idx], color='black', linestyle='--')
        axs[3].set_ylim(0, 1)
        axs[3].set_ylabel("R")
        plt.xlabel("Time (ns)")
        if self.cancel and self.echo:
            plt.suptitle("CR Echo Cancel Hamiltonian Tomography")
        elif self.cancel:
            plt.suptitle("CR Cancel Hamiltonian Tomography")
        elif self.echo:
            plt.suptitle("CR Echo Hamiltonian Tomography")
        else:
            plt.suptitle("CR Hamiltonian Tomography")

        plt.tight_layout()
        dirpath, filename = os.path.split(self.iname)
        iPATH = os.path.join(dirpath, "Ham_tomo", filename)
        plt.savefig(iPATH[:-4], bbox_inches='tight')
        plt.close(fig)
        meas_results.update({"R": R})
        self.data.update({
            "qubit_dynamics": meas_results,
            "interaction_rates": interaction_rates
        })

        for key, data in interaction_rates.items():
            print(f"{key}: {data[0]: 0.4f} +- {data[1]: 0.4f} MHz")

            Omega_ground, Omega_ground_error = calculate_Omega_MHz(popt_ground, pcov_ground)
            Omega_excited, Omega_excited_error = calculate_Omega_MHz(popt_excited, pcov_excited)
        print(f"Omega_ground: {Omega_ground: 0.4f} +- {Omega_ground_error: 0.4f} MHz")
        print(f"Omega_excited: {Omega_excited: 0.4f} +- {Omega_excited_error: 0.4f} MHz")

    def sweep_fit(self, data, p0s):
        """
        Sweep a list of guesses and fidn best fit to data
        """
        t_vec = self.t_vec
        best_fit = None
        best_fit_err = None
        for p0 in p0s:
            try:
                popt, pcov, infodict, _, ier = curve_fit(combined_func, t_vec, data, p0=p0,
                                                                  maxfev=10000, full_output=True)
                # extract residuals of fit
                fit_error = np.sum(infodict['fvec'] ** 2)
                if best_fit == None:
                    best_fit = (popt, pcov)
                    best_fit_err = fit_error
                if best_fit_err > fit_error:
                    best_fit = (popt, pcov)
                    best_fit_err = fit_error
            except:
               pass
        print(f"Best fit error: {best_fit_err}")
        return best_fit

    def fit_hamiltonian_tomography(self, data_dict):
        """
        Fit data to Hamiltonian tomography data
        """
        X0 = data_dict['<X>t']['qc = 0']
        X1 = data_dict['<X>t']['qc = 1']
        Y0 = data_dict['<Y>t']['qc = 0']
        Y1 = data_dict['<Y>t']['qc = 1']
        Z0 = data_dict['<Z>t']['qc = 0']
        Z1 = data_dict['<Z>t']['qc = 1']

        data_ground = np.concatenate([X0, Y0, Z0])
        guesses_ground = self.generate_fit_guesses(X0, Y0, Z0)
        popt_ground, pcov_ground = self.sweep_fit(data_ground, guesses_ground)

        data_excited = np.concatenate([X1, Y1, Z1])
        guesses_excited = self.generate_fit_guesses(X1, Y1, Z1)
        popt_excited, pcov_excited = self.sweep_fit(data_excited, guesses_excited)
        return popt_ground, pcov_ground, popt_excited, pcov_excited

    def generate_fit_guesses(self, X, Y, Z):
        """
        Calculate a range of initial conditions for the fit function
        """

        Z_freq = fft_expectation_values(self.t_vec, Z)
        freq_list = [Z_freq]
        if np.max(Y) - np.min(Y) > 0.2:
            Y_freq = fft_expectation_values(self.t_vec, Y)
            if abs(Y_freq) > 0:
                freq_list.append(Y_freq)
        if np.max(X)-np.min(X) > 0.2:
            X_freq = fft_expectation_values(self.t_vec, X)
            if abs(X_freq) > 0:
                freq_list.append(X_freq)
        Omega_guess = 2*np.pi * np.average(freq_list)
        print(f"Omega_guess: {Omega_guess}")

        theta = np.arccos(np.sqrt((np.max(Z) - np.min(Z))/2))

        toff_guess = 2*np.sqrt(2 * np.pi) * self.meta_dict['qc']['ZX45_len'] / 5
        b_guess = 0
        df = 1 / ((self.t_vec[1] - self.t_vec[0]) * len(self.t_vec))
        p0 = []

        # Empirical Guess in low frequency case
        if Omega_guess < df:
            p0.append((Omega_guess, Omega_guess, 0, toff_guess, b_guess))
        else:
            for Omega in [Omega_guess, Omega_guess - df/2, Omega_guess + df/2]:
                # for theta in np.linspace(0.5*theta, 1.5*theta, 3):
                    for phi in np.linspace(-np.pi, np.pi, 30):
                        px_guess = Omega * np.cos(theta) * np.cos(phi)
                        py_guess = Omega * np.cos(theta) * np.sin(phi)
                        pz_guess = Omega * np.sin(theta)
                        p0.append((px_guess, py_guess, pz_guess, toff_guess, b_guess))
        return p0

def fft_expectation_values(t_vec, data):
    N = len(data)
    dt = t_vec[1]-t_vec[0]
    freqs = fftfreq(N, d=dt)
    signal_zero_mean = data - np.mean(data)
    window = np.hanning(N)

    fft_result = fft(signal_zero_mean*window)
    fft_mag = np.abs(fft_result[:N//2])
    max_fft_mag = np.argmax(fft_mag)
    print(np.max(fft_mag))
    if np.max(fft_mag) > 1:
        return freqs[max_fft_mag]
    else:
        return 0
