from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiChevronTimedSS(ExperimentClass):
    """
    Rabi chevron (pulse duration vs frequency) with single-shot readout.
    t_vec: pulse durations in clock cycles (4 ns each)
    f_vec: frequency detunings in Hz (integers)
    """
    def __init__(self, element, t_vec, f_vec, shots, calib_params,
                 outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)

        self.element = element
        self.t_vec = t_vec
        self.f_vec = f_vec
        self.shots = shots
        self.calib_params = calib_params

        self.data = {
            'qubit': self.element,
            'shots': self.shots,
            't_vec': self.t_vec,
            'f_vec': self.f_vec,
            'meta_dict': self.meta_dict,
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
        shots = self.shots
        ss_thresh = self.calib_params['threshold']
        ground_confidence_threshold = self.calib_params['ground_threshold']
        IF_freq = self.meta_dict['IF_freq']

        with program() as prog:
            n = declare(int)
            df = declare(int)
            t = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(df, self.f_vec):
                    with for_(*from_array(t, self.t_vec)):
                        update_frequency(qubit, IF_freq + df)
                        play('X180', qubit, duration=t)
                        align(qubit, resonator)

                        # SS Readout
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, ss_stream)
                        align(qubit, resonator)

                        # Active reset
                        update_frequency(qubit, IF_freq)
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play('X180', qubit)
                                align(qubit, resonator)
                            I, Q = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                            wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                # Result shape: (len(f_vec), len(t_vec))
                ss_stream.buffer(len(self.t_vec)).buffer(len(self.f_vec)).average().save("ss")

        return prog

    def analyze(self, job, qm):
        results = fetching_tool(job, data_list=["ss"], mode="live")
        t_ns = self.t_vec  # clock cycles → ns

        matplotlib.use("qtagg", force=True)
        dummy = np.zeros((len(self.f_vec), len(self.t_vec)))
        fig, (ax_main, ax_line) = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 8))

        # Chevron pcolormesh
        im = ax_main.pcolormesh(t_ns, self.f_vec / u.MHz, dummy, shading='nearest', vmin=0, vmax=1)
        ax_main.set_xlabel('Pulse duration (ns)')
        ax_main.set_ylabel('Frequency detuning (MHz)')
        fig.colorbar(im, ax=ax_main, label='P(excited)')

        # Linecut at best frequency
        (lc_line,) = ax_line.plot(t_ns, np.zeros(len(self.t_vec)), 'o-', ms=3, lw=1.5)
        ax_line.axhline(0.5, color='gray', linestyle='--', lw=1)
        ax_line.set_xlabel('Pulse duration (ns)')
        ax_line.set_ylabel('P(excited)')
        ax_line.set_xlim(t_ns[0], t_ns[-1])
        ax_line.set_ylim(-0.05, 1.05)
        ax_line.grid(alpha=0.3)

        fig.suptitle(f"Timed Rabi {self.element}  "
                     f"{(self.meta_dict['q_LO']['LO_freq'] + self.meta_dict['IF_freq']) / u.MHz:.3f} MHz")
        interrupt_on_close(fig, job)

        ss_array = dummy.copy()
        while results.is_processing():
            ss_array = results.fetch_all()[0]
            if ss_array.shape != (len(self.f_vec), len(self.t_vec)):
                continue
            im.set_array(ss_array.ravel())
            im.set_clim(ss_array.min(), ss_array.max())
            best_f_idx = int(np.argmax(np.max(ss_array, axis=1)))
            lc_line.set_ydata(ss_array[best_f_idx])
            ax_line.set_title(f'Linecut at {self.f_vec[best_f_idx] / u.MHz:.2f} MHz detuning')
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        self.data.update({'ss_data': ss_array})
        qm.close()
        fig.savefig(self.iname, bbox_inches='tight')
        plt.close(fig)
        plt.ioff()