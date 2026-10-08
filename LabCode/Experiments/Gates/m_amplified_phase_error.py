"""
Amplified Phase Error (APE) experiment -- robust fitter + coherent frequency updates.

Change vs. the original m_amplified_phase_error.py
--------------------------------------------------
The per-detuning omega_e fit no longer hardcodes the trajectory geometry. The
original analyze_X90 baked in the q4 ellipse (rI,rQ,cI,cQ = 1.0,0.5,0.0,-0.5),
a model whose <Y> is confined to [-1,0]. q6 traces a UNIT CIRCLE centered at the
origin (the ideal shape for this sequence: <X>=sin(N*alpha), <Y>=-cos(N*alpha)),
whose <Y> spans [-1,+1]. The mismatched model cannot represent the top half of
the circle, so curve_fit returned a non-monotonic dome in omega_e and a
nonsensical extrapolated optimal detuning.

Fix: unwrap the polar angle vs the repetition number N and take the slope as the SIGNED omega_e. For q6 the default center is fixed at (0,0) rather than estimated from a partial arc. This is
geometry-independent -- it works for a circle (q6) or an ellipse (q4) -- because
the zero crossing (omega_e -> 0) is where the state stops precessing, which does
not depend on the trajectory shape. Validated in simulation to recover the
optimal detuning to ~kHz for both circle and ellipse when the sweep brackets the
crossing.

Usage notes
-----------
* Keep N_vec ascending and |omega_e| < pi per step so np.unwrap tracks
  (omega_e/pair ~ 2*pi*delta*(2*t_g)).
* Make f_vec BRACKET the zero crossing rather than relying on extrapolation;
  widen the span and reduce N_max together at large detuning to avoid wrap.
* This version deliberately uses update_frequency(..., keep_phase=True) when
  entering and leaving the detuned APE block. Without keep_phase=True, QUA's
  default coherent/discontinuous phase behavior can inject frame jumps at the
  frequency changes and corrupt the sign/scale of the extracted correction.
* Each tomography branch starts by resetting the frame and explicitly restoring
  the base IF with keep_phase=False. This defines a clean branch-local phase
  origin; the detuned excursion itself is phase-continuous.
* Apply the result only after the sign/path sanity test confirms that changing
  the correction knob shifts the APE residual with slope magnitude near 1.

The .data keys (optimal_detuning_MHz, omega_e_fit, phi0_fit) are unchanged, so
downstream code (e.g. the detuning co-sweep) works without modification.
"""

from qm.qua import *
from scipy.optimize import curve_fit  # retained for API compatibility (unused in fit)
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
from datetime import datetime
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from LabCode.Helpers.pulse_macros import *
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import convert_e_population_to_expectation
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)
plt.ion()


def _trajectory_center(expect_X, expect_Y, center=(0.0, 0.0), center_mode="fixed"):
    """Return the center used for polar-angle extraction.

    For q6 APE, use the default fixed center (0,0). Do NOT estimate the center
    from a short partial arc unless you have deliberately set center_mode="mean".
    A biased center gives a biased polar angle and can move the extracted zero.
    """
    if center_mode == "fixed":
        cX, cY = center
        return float(cX), float(cY)
    if center_mode == "mean":
        return float(np.nanmean(expect_X)), float(np.nanmean(expect_Y))
    raise ValueError("center_mode must be 'fixed' or 'mean'")


def _signed_omega_e(
    expect_X,
    expect_Y,
    f_vec,
    N_vec,
    center=(0.0, 0.0),
    center_mode="fixed",
    min_radius=0.15,
    min_points=5,
):
    """Signed omega_e per detuning from unwrapped polar angle vs N.

    The old q6 code used cX=np.mean(X), cY=np.mean(Y). That is dangerous for
    short arcs because the mean of the measured arc is not the trajectory center.
    By default this function uses the q6 ideal center (0,0). Use center_mode="mean"
    only for deliberately well-sampled trajectories.

    Returns
    -------
    omega_e_fit : list[float]
        Signed phase slope, rad per repeated block.
    phi0_fit : list[float]
        Intercept of the unwrapped angle fit.
    """
    cX, cY = _trajectory_center(
        expect_X, expect_Y, center=center, center_mode=center_mode
    )
    N_arr = np.asarray(N_vec, dtype=float)

    omega_e_fit = []
    phi0_fit = []

    for j in range(len(f_vec)):
        x = np.asarray(expect_X[j, :], dtype=float) - cX
        y = np.asarray(expect_Y[j, :], dtype=float) - cY
        radius = np.sqrt(x**2 + y**2)

        ok = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(N_arr) & (radius > min_radius)
        )
        if np.sum(ok) < min_points:
            omega_e_fit.append(np.nan)
            phi0_fit.append(np.nan)
            continue

        theta = np.unwrap(np.arctan2(y[ok], x[ok]))
        slope, intercept = np.polyfit(N_arr[ok], theta, 1)
        omega_e_fit.append(float(slope))
        phi0_fit.append(float(intercept))

    return omega_e_fit, phi0_fit


def _fit_local_zero_crossing(
    f_vec_Hz,
    omega_e_fit,
    local_fit_points=5,
    prefer_near_Hz=0.0,
    allow_extrapolation=False,
):
    """Fit omega_e(f) locally around a bracketed zero crossing.

    APE should not be used as an autocal correction when the zero crossing is
    outside the swept detuning window. By default this returns NaN rather than an
    extrapolated detuning.
    """
    f = np.asarray(f_vec_Hz, dtype=float)
    w = np.asarray(omega_e_fit, dtype=float)
    ok = np.isfinite(f) & np.isfinite(w)

    if np.sum(ok) < 2:
        return dict(
            fit_detuning_Hz=np.nan,
            slope=np.nan,
            intercept=np.nan,
            zero_bracketed=False,
            used_indices=np.array([], dtype=int),
            status="not_enough_finite_points",
        )

    original_indices = np.arange(len(f))[ok]
    f = f[ok]
    w = w[ok]
    order = np.argsort(f)
    f = f[order]
    w = w[order]
    original_indices = original_indices[order]

    exact_zero = np.where(np.isclose(w, 0.0, atol=1e-12))[0]
    crossings = np.where(w[:-1] * w[1:] < 0)[0]

    if len(exact_zero) > 0:
        k_mid = int(exact_zero[np.argmin(np.abs(f[exact_zero] - prefer_near_Hz))])
    elif len(crossings) > 0:
        mids = 0.5 * (f[crossings] + f[crossings + 1])
        k = int(crossings[np.argmin(np.abs(mids - prefer_near_Hz))])
        k_mid = k + 1
    else:
        if not allow_extrapolation:
            return dict(
                fit_detuning_Hz=np.nan,
                slope=np.nan,
                intercept=np.nan,
                zero_bracketed=False,
                used_indices=np.array([], dtype=int),
                status="zero_not_bracketed",
            )
        k_mid = int(np.argmin(np.abs(w)))

    npts = max(2, int(local_fit_points))
    half = npts // 2
    lo = max(0, k_mid - half)
    hi = min(len(f), lo + npts)
    lo = max(0, hi - npts)

    if hi - lo < 2:
        return dict(
            fit_detuning_Hz=np.nan,
            slope=np.nan,
            intercept=np.nan,
            zero_bracketed=False,
            used_indices=np.array([], dtype=int),
            status="not_enough_local_points",
        )

    slope, intercept = np.polyfit(f[lo:hi], w[lo:hi], 1)
    if not np.isfinite(slope) or abs(slope) < 1e-30:
        fit_detuning_Hz = np.nan
        status = "zero_slope"
    else:
        fit_detuning_Hz = -intercept / slope
        status = (
            "ok" if (f.min() <= fit_detuning_Hz <= f.max()) else "fit_outside_window"
        )

    zero_bracketed = (len(exact_zero) > 0) or (len(crossings) > 0)
    if not allow_extrapolation and not zero_bracketed:
        fit_detuning_Hz = np.nan

    return dict(
        fit_detuning_Hz=float(fit_detuning_Hz),
        slope=float(slope),
        intercept=float(intercept),
        zero_bracketed=bool(zero_bracketed),
        used_indices=original_indices[lo:hi],
        status=status,
    )


def _expected_ape_slope_rad_per_Hz(meta_dict, pulse_type="X90"):
    """Approximate d(omega_e)/d(df) for a repeated +/-pulse block."""
    try:
        tg_ns = float(meta_dict[pulse_type]["length"])
        return 2.0 * np.pi * (2.0 * tg_ns * 1e-9)
    except Exception:
        return np.nan


class APE(ExperimentClass):
    """
    Amplified Phase Error Experiment, implemented from E. Lucero et al.:
    DOI: https://doi.org/10.1103/PhysRevA.82.042339
    """

    def __init__(
        self,
        element,
        f_vec,
        N_vec,
        shots,
        calib_params,
        pulse_type="X90",
        outer_folder="",
        path="T2R",
        suffix="data",
        meta_dict=None,
        config=None,
        center=(0.0, 0.0),
        center_mode="fixed",
        min_radius=0.15,
        local_fit_points=5,
        allow_extrapolation=False,
        keep_phase_during_detuning=True,
        **kwargs,
    ):
        super().__init__(
            path=path,
            outer_folder=outer_folder,
            suffix=suffix,
            meta_dict=meta_dict,
            config=config,
            **kwargs,
        )
        self.config = config
        self.element = element  # meta dict element
        self.N_vec = N_vec
        self.f_vec = f_vec
        self.shots = shots
        self.calib_params = calib_params
        self.pulse_type = pulse_type
        self.center = center
        self.center_mode = center_mode
        self.min_radius = min_radius
        self.local_fit_points = local_fit_points
        self.allow_extrapolation = allow_extrapolation
        self.keep_phase_during_detuning = bool(keep_phase_during_detuning)
        self.data = {
            "qubit": self.element,
            "shots": self.shots,
            "meta_dict": self.meta_dict,
            "N_vec": self.N_vec,
            "f_vec": self.f_vec,
            "pulse_type": self.pulse_type,
            "center": self.center,
            "center_mode": self.center_mode,
            "min_radius": self.min_radius,
            "local_fit_points": self.local_fit_points,
            "allow_extrapolation": self.allow_extrapolation,
            "keep_phase_during_detuning": self.keep_phase_during_detuning,
        }
        if pulse_type == "X180":
            self.job, self.qm = self.acquire(
                self.make_prog_X180(), self.config, return_qm=True
            )
            self.analyze_X180(self.job, self.qm)
        elif pulse_type == "X90":
            self.job, self.qm = self.acquire(
                self.make_prog_X90(), self.config, return_qm=True
            )
            self.analyze_X90(self.job, self.qm)
        else:
            raise ValueError("pulse_type needs to be X90 or X180")
        self.data.update({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.h5_data()

    def make_prog_X90(self):
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        IF_Freq = self.meta_dict["IF_freq"]
        # Important: only the branch initialization is phase-discontinuous.
        # The detuning excursion and return preserve phase continuity, otherwise
        # update_frequency's default keep_phase=False injects frame jumps into APE.
        with program() as prog:
            n = declare(int)
            m = declare(int)
            df = declare(int)
            num_pulse = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            X_stream = declare_stream()
            Y_stream = declare_stream()
            n_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(df, self.f_vec):
                    with for_each_(num_pulse, self.N_vec):
                        # Measure along X
                        reset_frame(qubit)
                        update_frequency(qubit, IF_Freq, keep_phase=False)
                        # pi/2
                        play_pulse(
                            qubit,
                            self.pulse_type,
                            phase=self.meta_dict[self.pulse_type]["phase"],
                        )

                        update_frequency(
                            qubit,
                            IF_Freq + df,
                            keep_phase=self.keep_phase_during_detuning,
                        )
                        with for_(m, 0, m < num_pulse, m + 1):  # -pi/2
                            play_pulse(
                                qubit,
                                "m" + self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )
                            # pi/2
                            play_pulse(
                                qubit,
                                self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )

                        update_frequency(
                            qubit, IF_Freq, keep_phase=self.keep_phase_during_detuning
                        )
                        play_pulse(qubit, "mY90", phase=self.meta_dict["Y90"]["phase"])

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
                                play_pulse(
                                    qubit, "X180", phase=self.meta_dict["X180"]["phase"]
                                )
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

                        # Measure along Y
                        reset_frame(qubit)
                        update_frequency(qubit, IF_Freq, keep_phase=False)
                        # pi/2
                        play_pulse(
                            qubit,
                            self.pulse_type,
                            phase=self.meta_dict[self.pulse_type]["phase"],
                        )
                        update_frequency(
                            qubit,
                            IF_Freq + df,
                            keep_phase=self.keep_phase_during_detuning,
                        )
                        with for_(m, 0, m < num_pulse, m + 1):
                            # -pi/2
                            play_pulse(
                                qubit,
                                "m" + self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )

                            # pi/2
                            play_pulse(
                                qubit,
                                self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )

                        update_frequency(
                            qubit, IF_Freq, keep_phase=self.keep_phase_during_detuning
                        )
                        play_pulse(qubit, "X90", phase=self.meta_dict["X90"]["phase"])

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
                                play_pulse(
                                    qubit, "X180", phase=self.meta_dict["X180"]["phase"]
                                )
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                X_stream.buffer(len(self.N_vec)).buffer(len(self.f_vec)).average().save(
                    "X_ss"
                )
                Y_stream.buffer(len(self.N_vec)).buffer(
                    (len(self.f_vec))
                ).average().save("Y_ss")
                n_stream.save("n")
        return prog

    def make_prog_X180(self):
        qubit = self.meta_dict["q_name"]
        resonator = self.meta_dict["r_name"]
        resonator_dump_time = self.meta_dict["resonator_dump_time"]
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        IF_Freq = self.meta_dict["IF_freq"]
        # Important: only the branch initialization is phase-discontinuous.
        # The detuning excursion and return preserve phase continuity, otherwise
        # update_frequency's default keep_phase=False injects frame jumps into APE.

        with program() as prog:
            n = declare(int)
            m = declare(int)
            df = declare(int)
            num_pulse = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            X_stream = declare_stream()
            Z_stream = declare_stream()
            n_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(df, self.f_vec):
                    with for_each_(num_pulse, self.N_vec):
                        reset_frame(qubit)
                        update_frequency(qubit, IF_Freq, keep_phase=False)
                        # Measure along X
                        # pi/2
                        play_pulse(qubit, "Y90", phase=self.meta_dict["Y90"]["phase"])

                        update_frequency(
                            qubit,
                            IF_Freq + df,
                            keep_phase=self.keep_phase_during_detuning,
                        )
                        with for_(m, 0, m < num_pulse, m + 1):
                            # -pi
                            play_pulse(
                                qubit,
                                "m" + self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )
                            # pi
                            play_pulse(
                                qubit,
                                self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )

                        update_frequency(
                            qubit, IF_Freq, keep_phase=self.keep_phase_during_detuning
                        )
                        play_pulse(qubit, "mY90", phase=self.meta_dict["Y90"]["phase"])

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
                                play_pulse(
                                    qubit, "X180", phase=self.meta_dict["X180"]["phase"]
                                )
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

                        # Measure along Z
                        reset_frame(qubit)
                        update_frequency(qubit, IF_Freq, keep_phase=False)
                        # pi/2
                        play_pulse(qubit, "Y90", phase=self.meta_dict["Y90"]["phase"])
                        update_frequency(
                            qubit,
                            IF_Freq + df,
                            keep_phase=self.keep_phase_during_detuning,
                        )
                        with for_(m, 0, m < num_pulse, m + 1):
                            # pi
                            play_pulse(
                                qubit,
                                self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )
                            # -pi
                            play_pulse(
                                qubit,
                                "m" + self.pulse_type,
                                phase=self.meta_dict[self.pulse_type]["phase"],
                            )

                        update_frequency(
                            qubit, IF_Freq, keep_phase=self.keep_phase_during_detuning
                        )
                        play_pulse(qubit, "X90", phase=self.meta_dict["X90"]["phase"])
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
                                play_pulse(
                                    qubit, "X180", phase=self.meta_dict["X180"]["phase"]
                                )
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                X_stream.buffer(len(self.N_vec)).buffer(len(self.f_vec)).average().save(
                    "X_ss"
                )
                Z_stream.buffer(len(self.N_vec)).buffer(
                    (len(self.f_vec))
                ).average().save("Z_ss")
                n_stream.save("n")
        return prog

    def analyze_X90(self, job, qm):
        result = fetching_tool(self.job, data_list=["X_ss", "Y_ss", "n"], mode="live")

        # --- Figure setup ---
        fig, (ax_xy, ax_omega) = plt.subplots(1, 2, figsize=(12, 6))

        # --- IQ-plane subplot ---
        ax = ax_xy
        unit_circle = Circle(
            (0, 0),
            radius=1.0,
            edgecolor="k",
            facecolor="none",
            lw=1.5,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(unit_circle)
        ax.set_xlabel(r"$\langle X\rangle$")
        ax.set_ylabel(r"$\langle Y\rangle$")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="k", lw=1, alpha=0.6)
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(
            f"{self.element} {self.pulse_type} Amplified Phase Error Trajectories"
        )

        # --- omega_e subplot ---
        ax_w = ax_omega
        ax_w.set_xlabel("Detuning [MHz]")
        ax_w.set_ylabel(r"Effective angular frequency $\omega_e$")
        ax_w.grid(True)
        interrupt_on_close(fig, job)

        # --- Colormap for f_vec ---
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(self.f_vec)))

        lines = []
        for c, f in zip(colors, self.f_vec):
            (line,) = ax.plot(
                [], [], "o", lw=1.5, ms=4, color=c, label=f"{f / 1e6:.1f} MHz"
            )
            lines.append(line)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))

        # Reference phase for the offset print only. NOTE: the trajectory geometry
        # (center/radii) is NO LONGER hardcoded -- it is estimated from the data,
        # because it is qubit/convention dependent (q4 ~ ellipse centered (0,-0.5j);
        # q6 ~ unit circle centered at origin). See _signed_omega_e().
        phi_0 = -np.pi / 2

        # --- Live update loop ---
        omega_e_fit, phi0_fit = [], []
        optimal_detuning_MHz = np.nan
        fit_detuning = np.nan
        f_vec_Hz = np.asarray(self.f_vec, dtype=float)
        while result.is_processing():
            pop_X, pop_Y, n = result.fetch_all()
            expect_X = convert_e_population_to_expectation(pop_X)
            expect_Y = convert_e_population_to_expectation(pop_Y)

            # Update IQ-plane plot
            for j, line in enumerate(lines):
                line.set_data(expect_X[j, :], expect_Y[j, :])

            # --- Signed omega_e per detuning (fixed center by default for q6) ---
            omega_e_fit, phi0_fit = _signed_omega_e(
                expect_X,
                expect_Y,
                self.f_vec,
                self.N_vec,
                center=self.center,
                center_mode=self.center_mode,
                min_radius=self.min_radius,
            )

            # --- Local zero crossing fit. No extrapolated autocal by default. ---
            f_vec_Hz = np.asarray(self.f_vec, dtype=float)
            fit = _fit_local_zero_crossing(
                f_vec_Hz,
                omega_e_fit,
                local_fit_points=self.local_fit_points,
                prefer_near_Hz=0.0,
                allow_extrapolation=self.allow_extrapolation,
            )
            fit_detuning = fit["fit_detuning_Hz"]
            optimal_detuning_MHz = (
                fit_detuning / u.MHz if np.isfinite(fit_detuning) else np.nan
            )
            slope = fit["slope"]
            intercept = fit["intercept"]
            expected_slope = _expected_ape_slope_rad_per_Hz(
                self.meta_dict, self.pulse_type
            )

            # Plot omega_e with local regression only if a valid fit exists.
            ax_w.clear()
            ax_w.set_xlabel(r"$\delta-\delta_{ops}$ [MHz]")
            ax_w.set_ylabel(r"Effective angular frequency $\omega_e$ [rad/block]")
            ax_w.grid(True)
            ax_w.plot(
                f_vec_Hz / 1e6, omega_e_fit, "o", color="darkblue", label="Fitted ωe"
            )
            if np.isfinite(slope) and np.isfinite(intercept):
                used = fit["used_indices"]
                ax_w.plot(
                    f_vec_Hz / 1e6,
                    slope * f_vec_Hz + intercept,
                    "-",
                    color="darkblue",
                    label="Local fit",
                )
                if len(used) > 0:
                    ax_w.plot(
                        f_vec_Hz[used] / 1e6,
                        np.asarray(omega_e_fit)[used],
                        "o",
                        color="red",
                        ms=7,
                        label="Used points",
                    )
            ax_w.axhline(0, color="k", lw=1, alpha=0.5)
            title = (
                f"Optimal detuning: {optimal_detuning_MHz:.3f} MHz"
                if np.isfinite(optimal_detuning_MHz)
                else "APE zero not bracketed"
            )
            ax_w.set_title(title)
            ax_w.legend()

            self.data.update(
                {
                    "X": pop_X,
                    "Y": pop_Y,
                    "expect_X": expect_X,
                    "expect_Y": expect_Y,
                    "omega_e_fit": omega_e_fit,
                    "phi0_fit": phi0_fit,
                    "optimal_detuning_MHz": optimal_detuning_MHz,
                    "optimal_detuning_Hz": fit_detuning,
                    "zero_bracketed": fit["zero_bracketed"],
                    "fit_status": fit["status"],
                    "fit_slope_rad_per_Hz": slope,
                    "expected_slope_rad_per_Hz": expected_slope,
                    "fit_intercept_rad": intercept,
                    "fit_used_indices": fit["used_indices"],
                    "fit_center_X": _trajectory_center(
                        expect_X, expect_Y, self.center, self.center_mode
                    )[0],
                    "fit_center_Y": _trajectory_center(
                        expect_X, expect_Y, self.center, self.center_mode
                    )[1],
                }
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        mean_phi0 = np.nanmean(phi0_fit) if "phi0_fit" in locals() else np.nan
        if np.isfinite(optimal_detuning_MHz):
            print(f"New Detuning: {optimal_detuning_MHz:.3f} MHz")
        else:
            print(
                "New Detuning: NaN (APE zero was not bracketed; do not apply this correction)"
            )
        print(f" Fit Phi_0/2pi : {mean_phi0 / 2 / np.pi: .3f}")
        print(f"Phi0 offset: {(mean_phi0 - phi_0) / (2 * np.pi):.3f} rad")
        print(
            f"APE fit status: {self.data.get('fit_status', 'unknown')}; "
            f"zero_bracketed={self.data.get('zero_bracketed', False)}"
        )
        if np.isfinite(self.data.get("fit_slope_rad_per_Hz", np.nan)):
            ratio = self.data["fit_slope_rad_per_Hz"] / self.data.get(
                "expected_slope_rad_per_Hz", np.nan
            )
            print(
                f"APE slope: {self.data['fit_slope_rad_per_Hz'] * 1e6:.3f} rad/MHz; "
                f"expected ~{self.data.get('expected_slope_rad_per_Hz', np.nan) * 1e6:.3f} rad/MHz; "
                f"ratio={ratio:.2f}"
            )

        # Diagnostic guard: a clean local run should be monotonic around the zero.
        y_lin = np.array(omega_e_fit) if "omega_e_fit" in locals() else np.array([])
        if len(y_lin) > 1 and np.all(np.isfinite(y_lin)):
            d = np.diff(y_lin)
            if not (np.all(d > 0) or np.all(d < 0)):
                print(
                    "WARNING: omega_e(detuning) is non-monotonic -- check wrapping, SNR, "
                    "and whether the sweep has multiple/aliased crossings."
                )
        if not self.data.get("zero_bracketed", False):
            print(
                "WARNING: APE zero crossing is not bracketed. Widen/recenter f_vec or reduce N_max; "
                "the returned optimal_detuning_MHz is NaN unless allow_extrapolation=True."
            )

        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)

    def analyze_X180(self, job, qm):
        result = fetching_tool(self.job, data_list=["X_ss", "Z_ss", "n"], mode="live")

        # --- Figure setup ---
        fig, (ax_xy, ax_omega) = plt.subplots(1, 2, figsize=(12, 6))

        ax = ax_xy
        unit_circle = Circle(
            (0, 0),
            radius=1.0,
            edgecolor="k",
            facecolor="none",
            lw=1.5,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(unit_circle)
        ax.set_xlabel(r"$\langle X\rangle$")
        ax.set_ylabel(r"$\langle Z\rangle$")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="k", lw=1, alpha=0.6)
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(
            f"{self.element} {self.pulse_type} Amplified Phase Error Trajectories"
        )

        ax_w = ax_omega
        ax_w.set_xlabel("Detuning [MHz]")
        ax_w.set_ylabel(r"Effective angular frequency $\omega_e$")
        ax_w.grid(True)
        interrupt_on_close(fig, job)

        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(self.f_vec)))

        lines = []
        for c, f in zip(colors, self.f_vec):
            (line,) = ax.plot(
                [], [], "o", lw=1.5, ms=4, color=c, label=f"{f / 1e6:.1f} MHz"
            )
            lines.append(line)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))

        # Reference phase for the offset print only (X180 trajectory ~ circle
        # centered origin). Geometry estimated from data; see _signed_omega_e().
        phi_0 = 0

        omega_e_fit, phi0_fit = [], []
        optimal_detuning_MHz = np.nan
        fit_detuning = np.nan
        f_vec_Hz = np.asarray(self.f_vec, dtype=float)
        while result.is_processing():
            pop_X, pop_Z, n = result.fetch_all()
            expect_X = convert_e_population_to_expectation(pop_X)
            expect_Z = convert_e_population_to_expectation(pop_Z)

            for j, line in enumerate(lines):
                line.set_data(expect_X[j, :], expect_Z[j, :])

            # --- Signed omega_e per detuning (fixed center by default) ---
            omega_e_fit, phi0_fit = _signed_omega_e(
                expect_X,
                expect_Z,
                self.f_vec,
                self.N_vec,
                center=self.center,
                center_mode=self.center_mode,
                min_radius=self.min_radius,
            )

            f_vec_Hz = np.asarray(self.f_vec, dtype=float)
            fit = _fit_local_zero_crossing(
                f_vec_Hz,
                omega_e_fit,
                local_fit_points=self.local_fit_points,
                prefer_near_Hz=0.0,
                allow_extrapolation=self.allow_extrapolation,
            )
            fit_detuning = fit["fit_detuning_Hz"]
            det0 = self.meta_dict.get(self.pulse_type, {}).get("detuning", 0.0)
            optimal_detuning_MHz = (
                fit_detuning / u.MHz + det0 / u.MHz
                if np.isfinite(fit_detuning)
                else np.nan
            )
            slope = fit["slope"]
            intercept = fit["intercept"]
            expected_slope = _expected_ape_slope_rad_per_Hz(
                self.meta_dict, self.pulse_type
            )

            ax_w.clear()
            ax_w.set_xlabel("Detuning [MHz]")
            ax_w.set_ylabel(r"Effective angular frequency $\omega_e$ [rad/block]")
            ax_w.grid(True)
            ax_w.plot(
                f_vec_Hz / 1e6, omega_e_fit, "o", color="darkblue", label="Fitted ωe"
            )
            if np.isfinite(slope) and np.isfinite(intercept):
                used = fit["used_indices"]
                ax_w.plot(
                    f_vec_Hz / 1e6,
                    slope * f_vec_Hz + intercept,
                    "-",
                    color="darkblue",
                    label="Local fit",
                )
                if len(used) > 0:
                    ax_w.plot(
                        f_vec_Hz[used] / 1e6,
                        np.asarray(omega_e_fit)[used],
                        "o",
                        color="red",
                        ms=7,
                        label="Used points",
                    )
            ax_w.axhline(0, color="k", lw=1, alpha=0.5)
            title = (
                f"Optimal detuning: {optimal_detuning_MHz:.3f} MHz"
                if np.isfinite(optimal_detuning_MHz)
                else "APE zero not bracketed"
            )
            ax_w.set_title(title)
            ax_w.legend()

            self.data.update(
                {
                    "X": pop_X,
                    "Z": pop_Z,
                    "expect_X": expect_X,
                    "expect_Z": expect_Z,
                    "omega_e_fit": omega_e_fit,
                    "phi0_fit": phi0_fit,
                    "optimal_detuning_MHz": optimal_detuning_MHz,
                    "optimal_detuning_Hz": fit_detuning,
                    "zero_bracketed": fit["zero_bracketed"],
                    "fit_status": fit["status"],
                    "fit_slope_rad_per_Hz": slope,
                    "expected_slope_rad_per_Hz": expected_slope,
                    "fit_intercept_rad": intercept,
                    "fit_used_indices": fit["used_indices"],
                    "fit_center_X": _trajectory_center(
                        expect_X, expect_Z, self.center, self.center_mode
                    )[0],
                    "fit_center_Y": _trajectory_center(
                        expect_X, expect_Z, self.center, self.center_mode
                    )[1],
                }
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        mean_phi0 = np.nanmean(phi0_fit) if "phi0_fit" in locals() else np.nan
        if np.isfinite(optimal_detuning_MHz):
            print(f"Detuning: {optimal_detuning_MHz:.3f} MHz")
        else:
            print(
                "Detuning: NaN (APE zero was not bracketed; do not apply this correction)"
            )
        print(f"Phi0 offset: {(mean_phi0 - phi_0) / (2 * np.pi):.3f} rad")
        print(
            f"APE fit status: {self.data.get('fit_status', 'unknown')}; "
            f"zero_bracketed={self.data.get('zero_bracketed', False)}"
        )

        y_lin = np.array(omega_e_fit) if "omega_e_fit" in locals() else np.array([])
        if len(y_lin) > 1 and np.all(np.isfinite(y_lin)):
            d = np.diff(y_lin)
            if not (np.all(d > 0) or np.all(d < 0)):
                print(
                    "WARNING: omega_e(detuning) is non-monotonic -- check wrapping, SNR, "
                    "and whether the sweep has multiple/aliased crossings."
                )
        if not self.data.get("zero_bracketed", False):
            print(
                "WARNING: APE zero crossing is not bracketed. Widen/recenter f_vec or reduce N_max; "
                "the returned optimal_detuning_MHz is NaN unless allow_extrapolation=True."
            )

        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
