from qm.qua import *
import matplotlib
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from LabCode.Helpers.ss_helpers import find_state_1q_nL
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class ALE_Thermal(ExperimentClass):
    """
    Amplified Leakage Error Experiment — thermal reset variant.

    Same sequence as ALE but uses a thermalization wait for state preparation
    instead of measurement + post-selection:

        wait(resonator_dump_time) → (X Z(theta))^N → Measure |2>

    Faster per shot (no pre-measurement) but requires T1 >> experiment time
    to guarantee ground-state initialization.
    """

    def __init__(self, element, theta_vec, N_vec, shots, calib_params,
                 outer_folder='', path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element
        self.theta_vec = theta_vec
        self.N_vec = N_vec
        self.shots = shots
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "theta_vec": self.theta_vec,
            "N_vec": self.N_vec,
            'shots': self.shots,
            'meta_dict': self.meta_dict,
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict["reset_time"]
        shots = self.shots

        with program() as prog:
            n = declare(int)
            m = declare(int)
            theta = declare(fixed)
            num_pulse = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_result_stream = declare_stream()
            Q_result_stream = declare_stream()
            n_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(num_pulse, self.N_vec):
                    with for_each_(theta, self.theta_vec):
                        # Thermal reset
                        wait(reset_time * u.ns)

                        # ALE sequence
                        reset_frame(qubit)
                        with for_(m, 0, m < num_pulse, m + 1):
                            play('X90', qubit)
                            play('X90', qubit)
                            frame_rotation_2pi(theta, qubit)
                        align(qubit, resonator)

                        # Measurement
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_result_stream)
                        save(Q, Q_result_stream)
                        align(qubit, resonator)

            with stream_processing():
                n_stream.save("n")

                I_result_stream \
                    .buffer(len(self.theta_vec)) \
                    .buffer(len(self.N_vec)) \
                    .save("I_result")

                Q_result_stream \
                    .buffer(len(self.theta_vec)) \
                    .buffer(len(self.N_vec)) \
                    .save("Q_result")

        return prog

    def analyze(self, job, qm):
        matplotlib.use("QtAgg")
        plt.ion()

        result = fetching_tool(
            job,
            data_list=["I_result", "Q_result", "n"],
            mode="live",
        )

        length_theta = len(self.theta_vec)
        length_N = len(self.N_vec)

        # ---------------- Figure ----------------
        fig, (ax_img, ax_line) = plt.subplots(
            2, 1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        interrupt_on_close(fig, job)

        # -------- Heatmap --------
        data = np.zeros((length_N, length_theta))
        im = ax_img.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Reds",
            vmin=0,
            vmax=0.5,
        )

        cbar = plt.colorbar(im, ax=ax_img)
        cbar.set_label("|2⟩ state population")
        ax_img.set_ylabel("N")

        num_ticks = min(5, length_theta)
        theta_tick_idx = np.linspace(0, length_theta - 1, num_ticks, dtype=int)
        ax_img.set_xticks(theta_tick_idx)
        ax_img.set_xticklabels([f"{self.theta_vec[i]:.3f}" for i in theta_tick_idx])
        ax_img.set_xlabel("θ / 2π")

        num_ticks_N = min(5, length_N)
        N_tick_idx = np.linspace(0, length_N - 1, num_ticks_N, dtype=int)
        ax_img.set_yticks(N_tick_idx)
        ax_img.set_yticklabels([str(self.N_vec[i]) for i in N_tick_idx])

        # -------- Line plot (max over N) --------
        theta_vals = np.asarray(self.theta_vec)
        max_line, = ax_line.plot(
            theta_vals,
            np.zeros_like(theta_vals),
            lw=2,
            color="red",
        )

        peak_scatter = ax_line.scatter([], [], color="red", s=30, zorder=3)
        peak_texts = []

        ax_line.set_xlabel("θ / 2π")
        ax_line.set_ylabel("max P(|2⟩)")
        ax_line.set_ylim(0, 0.5)
        ax_line.grid(True, alpha=0.3)

        # -------- Accumulators --------
        sum_data = np.zeros((length_N, length_theta), dtype=np.float64)
        count_data = 0

        # -------- Plot throttling --------
        last_draw = time.time()
        draw_period = 0.2  # seconds
        last_iteration = -1

        while result.is_processing():
            I, Q, iteration = result.fetch_all()
            iteration = int(iteration)

            if iteration == last_iteration:
                plt.pause(0.02)
                continue
            last_iteration = iteration

            I = np.asarray(I)
            Q = np.asarray(Q)

            if I.shape != Q.shape:
                continue

            # Shape: (N, theta) → add fake shot axis for classifier
            state = find_state_1q_nL(I[None, ...], Q[None, ...], self.calib_params)[0]

            sum_data += (state == 2)
            count_data += 1

            print(f"\r{iteration}/{self.shots}", end="", flush=True)

            now = time.time()
            if now - last_draw > draw_period:
                data[:] = sum_data / max(count_data, 1)

                im.set_data(data)

                max_over_N = np.nanmax(data, axis=0)
                max_line.set_ydata(max_over_N)

                # Peaks above threshold with minimum index spacing
                threshold = 0.1
                peak_indices = np.where(max_over_N > threshold)[0]
                peak_values = max_over_N[peak_indices]

                if len(peak_values) > 0:
                    sort_idx = np.argsort(peak_values)[::-1]
                    peak_indices = peak_indices[sort_idx]
                    peak_values = peak_values[sort_idx]

                    selected_indices = []
                    selected_values = []
                    min_index_spacing = 5

                    for idx, val in zip(peak_indices, peak_values):
                        if all(abs(idx - i) >= min_index_spacing for i in selected_indices):
                            selected_indices.append(idx)
                            selected_values.append(val)
                        if len(selected_indices) >= 6:
                            break

                    peak_theta = theta_vals[selected_indices]
                    peak_vals = np.array(selected_values)
                else:
                    peak_theta = np.array([])
                    peak_vals = np.array([])

                max_line.set_ydata(max_over_N)
                peak_scatter.set_offsets(np.column_stack([peak_theta, peak_vals]) if len(peak_theta) else np.empty((0, 2)))

                for txt in peak_texts:
                    txt.remove()
                peak_texts.clear()

                for th, val in zip(peak_theta, peak_vals):
                    peak_texts.append(
                        ax_line.text(
                            th, val,
                            f"({th:.3f}, {val:.2f})",
                            fontsize=8,
                            color="red",
                            ha="center",
                            va="bottom",
                        )
                    )

                plt.pause(0.1)
                last_draw = now

        print("")
        qm.close()

        self.data["2_state_array"] = sum_data / max(count_data, 1)

        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
