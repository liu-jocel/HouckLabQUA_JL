from qm.qua import *
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("qtagg", force=True)
from datetime import datetime
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from LabCode.Helpers.pulse_macros import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class APE1DFreq(ExperimentClass):
    """
    Frequency Calibration with APE1D experiment
    """
    def __init__(self,element, phi_vec, f_vec, n_vec, shots, calib_params,outer_folder='',path='T2R', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.config = config
        self.element = element #meta dict element
        self.n_vec = n_vec
        self.f_vec = f_vec
        self.shots = shots
        self.phi_vec = phi_vec
        self.calib_params = calib_params
        self.data = {
            "qubit": self.element,
            "n_vec": self.n_vec,
            'shots': self.shots,
            'phi_vec': self.phi_vec,
            'meta_dict': self.meta_dict
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        IF_Freq = self.meta_dict['IF_freq'] + self.meta_dict['detuning']

        with program() as prog:
            n = declare(int)
            m = declare(int)
            m_max = declare(int)
            df = declare(int)
            p = declare(fixed)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(df, self.f_vec):
                    with for_each_(m_max, self.n_vec):
                        with for_each_(p, self.phi_vec):
                            reset_frame(qubit)
                            update_frequency(qubit, IF_Freq + df)
                            play("X90", qubit)
                            with for_(m, 0, m < m_max, m + 1):
                                play("X90", qubit)
                                wait(self.meta_dict['X90']['length']*u.ns, qubit)
                                play("mX90", qubit)
                            frame_rotation_2pi(p, qubit)
                            play("X90", qubit)
                            update_frequency(qubit, IF_Freq)

                            # SS Readout
                            align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
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
                                    play_pulse(qubit, "X180",
                                               phase=self.meta_dict["X180"]['phase'])
                                    align(qubit, resonator)
                                I, _ = measure_1q(self.meta_dict, I, Q)
                                assign(I_reset, I)
                            wait(resonator_dump_time*u.ns, qubit)
            with stream_processing():
                ss_stream.buffer(len(self.phi_vec)).buffer(len(self.n_vec)).buffer(len(self.f_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):

        result = fetching_tool(self.job, data_list=["ss"], mode="live")

        # ---------- Dimensions ----------
        nN = len(self.n_vec)
        nF = len(self.f_vec)

        # ---------- Storage ----------
        period_mat = np.full((nF, nN), np.nan)
        phi0_mat = np.full((nF, nN), np.nan)
        amp_mat = np.full((nF, nN), np.nan)
        offset_mat = np.full((nF, nN), np.nan)

        # ---------- Figure setup (LIVE) ----------
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"{self.element} APE Frequency Calibration")

        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Phi_0")
        ax.grid(True)

        interrupt_on_close(fig, job)

        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, nN))

        # One line per n
        period_lines = []
        for i_n, (n, c) in enumerate(zip(self.n_vec, colors)):
            (line,) = ax.plot(
                self.f_vec / 1e6,
                np.full(nF, np.nan),
                "o-",
                color=c,
                lw=1.5,
                ms=4,
                label=f"n={n}"
            )
            period_lines.append(line)

        ax.legend(title="n", fontsize=9)

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        # ---------- Fit model ----------
        def cosine(x, A, period, phi0, C):
            return A * np.cos(2 * np.pi * (x / period + phi0)) + C

        # ---------- Live loop ----------
        while result.is_processing():

            ss = result.fetch_all()[0]

            if ss.size == 0 or ss.ndim != 3:
                plt.pause(0.05)
                continue

            # ss shape: (nF, nN, nPhi)
            for i_f in range(nF):
                for i_n in range(nN):

                    y = ss[i_f, i_n, :]

                    if np.all(y == 0):
                        continue

                    A0 = 0.5 * (np.max(y) - np.min(y))
                    C0 = np.mean(y)

                    try:
                        popt, _ = curve_fit(
                            cosine,
                            self.phi_vec,
                            y,
                            p0=[A0, 1.0, 0.0, C0],
                            maxfev=6000
                        )

                        A, period, phi0, C = popt

                        period_mat[i_f, i_n] = period
                        phi0_mat[i_f, i_n] = phi0
                        amp_mat[i_f, i_n] = A
                        offset_mat[i_f, i_n] = C

                    except RuntimeError:
                        continue

            # ---------- Live update ----------
            for i_n in range(nN):
                period_lines[i_n].set_ydata(phi0_mat[:, i_n])

            ax.relim()
            ax.autoscale_view()

            plt.pause(0.05)
            plt.pause(0.05)

        # ---------- Save ----------
        self.data.update({
            "period_mat": period_mat,
            "phi0_mat": phi0_mat,
            "amp_mat": amp_mat,
            "offset_mat": offset_mat,
            "n_vec": self.n_vec,
            "f_vec": self.f_vec,
        })

        print("done!")
        qm.close()
        plt.ioff()
        fig.savefig(self.iname.replace(".pkl", "_period_vs_detuning.png"),
                    bbox_inches="tight")
        plt.close(fig)


