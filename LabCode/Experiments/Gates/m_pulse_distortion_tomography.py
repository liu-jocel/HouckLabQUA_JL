from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from datetime import datetime
import matplotlib.pyplot as plt
plt.ion()
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

INVERSE_PULSE = {"X90": "mX90", "mX90": "X90", "Y90": "mY90", "mY90": "Y90"}
# Tomography pre-measurement rotation for each basis, using the already-calibrated
# pulse family (no separate pulse length/calibration): Z -> no pulse, X -> mY90, Y -> X90.
TOMO_BASES = [("Z", None), ("X", "mY90"), ("Y", "X90")]


class PulseDistortionTomography(ExperimentClass):
    """
    Repeats (pulse_type, inverse(pulse_type)) N times for every N in num_pulses_vec,
    then performs single-qubit state tomography (Z/X/Y) to reveal the accumulated
    distortion signature of back-to-back opposite pulses (e.g. pulse-to-pulse
    transient/memory effects) as a function of N.
    """
    def __init__(self, element, num_pulses_vec, shots, calib_params, pulse_type="X90",
                 outer_folder='', path='PulseDistTomo', suffix='data', meta_dict=None,
                 config=None, **kwargs):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                          meta_dict=meta_dict, config=config, **kwargs)
        self.config = config
        self.element = element
        self.num_pulses_vec = num_pulses_vec
        self.shots = shots
        self.calib_params = calib_params
        self.pulse_type = pulse_type.upper()
        self.inverse_pulse_type = INVERSE_PULSE[self.pulse_type]
        self.data = {
            "qubit": self.element,
            "shots": self.shots,
            "pulse_type": self.pulse_type,
            "num_pulses_vec": list(self.num_pulses_vec),
            "meta_dict": self.meta_dict,
        }

        self.job, self.qm = self.acquire(self.make_prog(), self.config, return_qm=True)
        self.analyze(self.job, self.qm)
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        resonator_dump_time = self.meta_dict['resonator_dump_time']
        shots = self.shots
        ss_thresh = self.calib_params["threshold"]
        ground_confidence_threshold = self.calib_params["ground_threshold"]
        pulse = self.pulse_type
        inv_pulse = self.inverse_pulse_type

        with program() as prog:
            n = declare(int)
            num_pulse = declare(int)
            m = declare(int)
            state = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_reset = declare(fixed)
            ss_stream_z = declare_stream()
            ss_stream_x = declare_stream()
            ss_stream_y = declare_stream()
            n_stream = declare_stream()
            streams = {"Z": ss_stream_z, "X": ss_stream_x, "Y": ss_stream_y}

            with for_(n, 0, n < shots, n + 1):
                save(n, n_stream)
                with for_each_(num_pulse, self.num_pulses_vec):
                    for basis_label, tomo_op in TOMO_BASES:
                        reset_frame(qubit)
                        align()
                        with for_(m, 0, m < num_pulse, m + 1):
                            play(pulse, qubit) # compose X180 of two X90s
                            play(pulse, qubit)
                            play(inv_pulse, qubit)  # compose mX180 of two mX90s
                            play(inv_pulse, qubit)
                        if tomo_op is not None:
                            play(tomo_op, qubit)

                        # SS Readout
                        align(qubit, resonator)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        assign(I_reset, I)
                        with if_(I > ss_thresh):
                            assign(state, 1)
                        with else_():
                            assign(state, 0)
                        save(state, streams[basis_label])
                        align(qubit, resonator)

                        # Active reset
                        with while_(I_reset > ground_confidence_threshold):
                            with if_(I_reset > ss_thresh):
                                play("X180", qubit)
                                align(qubit, resonator)
                            I, _ = measure_1q(self.meta_dict, I, Q)
                            assign(I_reset, I)
                        wait(resonator_dump_time * u.ns, qubit)

            with stream_processing():
                ss_stream_z.buffer(len(self.num_pulses_vec)).average().save("pop_Z")
                ss_stream_x.buffer(len(self.num_pulses_vec)).average().save("pop_X")
                ss_stream_y.buffer(len(self.num_pulses_vec)).average().save("pop_Y")
                n_stream.save("iteration")
        return prog

    def analyze(self, job, qm):
        result = fetching_tool(job, data_list=["pop_Z", "pop_X", "pop_Y", "iteration"], mode="live")

        fig, ax = plt.subplots()
        ax.set_xlabel("N  (number of "
                       f"{self.pulse_type}-{self.inverse_pulse_type} pairs)")
        ax.set_ylabel("Bloch component")
        ax.set_ylim(-1.05, 1.05)
        (lz,) = ax.plot(self.num_pulses_vec, [0] * len(self.num_pulses_vec), "o-", label="<Z>")
        (lx,) = ax.plot(self.num_pulses_vec, [0] * len(self.num_pulses_vec), "o-", label="<X>")
        (ly,) = ax.plot(self.num_pulses_vec, [0] * len(self.num_pulses_vec), "o-", label="<Y>")
        ax.legend()
        ax.set_title(f"{self.element}  {self.pulse_type}-{self.inverse_pulse_type} pulse-distortion tomography")
        interrupt_on_close(fig, job)

        pop_z = pop_x = pop_y = None
        while result.is_processing():
            pop_z, pop_x, pop_y, _ = result.fetch_all()

            # Same sign convention as SingleShot1Q: state=1 (I > threshold) is the
            # excited-state population, so <A> = 1 - 2*P(excited) for each basis A.
            bloch_z = 1 - 2 * pop_z
            bloch_x = 1 - 2 * pop_x
            bloch_y = 1 - 2 * pop_y
            self.data.update({
                'pop_Z': pop_z, 'pop_X': pop_x, 'pop_Y': pop_y,
                'bloch_Z': bloch_z, 'bloch_X': bloch_x, 'bloch_Y': bloch_y,
            })

            lz.set_data(self.num_pulses_vec, bloch_z)
            lx.set_data(self.num_pulses_vec, bloch_x)
            ly.set_data(self.num_pulses_vec, bloch_y)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

        qm.close()
        plt.ioff()
        fig.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
