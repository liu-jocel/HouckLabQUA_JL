import numpy as np
import time
import scipy
import math
from qm.qua import infinite_loop_, program, play
from qm import QuantumMachinesManager

# from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Config.make_meta_dict import make_meta_dict
from Drivers.signalhound_sa124 import SA124


class MixerTuner:
    """
    Use tune_lo or tune_sb to minimize local oscillator leakage and upper sideband leakage respectively. Both methods require the mode object as well as a method key - either "BF" (default) to run Brute-Force minimizer or "NM" to run Nelder-Mead minimizer. The BF minimizer allows more control over the minimization but is slower. The NM minimizer is faster but its inner workings cannot be finely controlled.

    Note that the BF minimizer is minimizing the amplitude level whereas the NM minimizer is minimizing the contrast between signal peak and floor (to lower than the 3dB threshold).
    """

    # parameters common to both Nelder-Mead and Brute-Force minimization
    threshold: float = 3.0  # in dBm
    ref_power: float = -40.0

    # Nelder-Mead (NM) parameters
    simplex: np.ndarray = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])
    maxiter: int = 100
    span: float = 2e6
    rbw: float = 50e3

    def __init__(self, sa: SA124) -> None:
        """ """
        self._sa = sa

        self._qm_job = None  # will be set and halted by the tune methods
        self._initial_contrast = None  # set during tuning check

    def _setup(self, element_name: str):
        """Play carrier frequency and intermediate frequency to mode and check current tuning"""
        self._qm = QM
        self._qm_job = self._qm.execute(self._get_qua_program(element_name))

    def _get_qua_program(self, element_name):
        """ """
        with program() as mixer_tuning:
            with infinite_loop_():
                play(OP_NAME, element_name)
        return mixer_tuning

    def tune_lo(self, element_name):
        """ """
        self._setup(element_name)

        is_tuned, center_idx, floor = self._check_tuning(element_name, "LO")
        i_offset, q_offset = None, None

        if is_tuned:
            print(f"LO already tuned to within {self.threshold} dBm!")
            self._qm_job.halt()
            return i_offset, q_offset

        print(f"Minimizing {element_name} LO leakage...")

        def objective_fn(offsets: tuple[float]) -> float:
            i_offset, q_offset = offsets[0], offsets[1]
            self._qm.set_output_dc_offset_by_element(element_name, "I", i_offset)
            self._qm.set_output_dc_offset_by_element(element_name, "Q", q_offset)
            contrast = self._get_contrast(center_idx, floor)
            print(f"Set I: {i_offset}, Q: {q_offset}. {contrast = }")
            return contrast

        result = self._minimize_nm(objective_fn, bounds=((-0.5, 0.5), (-0.5, 0.5)))
        if result is not None:
            i_offset, q_offset = result

        self._qm_job.halt()

        print(f"{element_name}")
        print(f"{i_offset = }")
        print(f"{q_offset = }")

        return i_offset, q_offset

    def tune_sb(self, element_name):
        """ """
        self._setup(element_name)

        is_tuned, center_idx, floor = self._check_tuning(element_name, "SB")
        g_offset, p_offset = None, None

        if is_tuned:
            print(f"SB already tuned to within {self.threshold} dBm!")
            self._qm_job.halt()
            return g_offset, p_offset

        print(f"Minimizing {element_name} SB leakage...")

        def objective_fn(offsets: tuple[float]) -> float:
            c_matrix = self._get_correction_matrix(*offsets)
            if any(x < -2 or x > 2 for x in c_matrix):
                print("WARNING: Found out of bounds value in c-matrix")
                return self._initial_contrast
            self._qm_job.set_element_correction(element_name, c_matrix)
            contrast = self._get_contrast(center_idx, floor)
            print(f"Set G: {offsets[0]}, P: {offsets[1]}. {contrast = }")
            return contrast

        result = self._minimize_nm(objective_fn, bounds=None)
        if result is not None:
            g_offset, p_offset = result

        self._qm_job.halt()

        print(f"{element_name}")
        print(f"{g_offset = }")
        print(f"{p_offset = }")

        return g_offset, p_offset

    def _get_correction_matrix(self, g: float, p: float) -> list[float]:
        """ """
        try:
            cos, sin = np.cos(p), np.sin(p)
            coefficient = 1 / ((1 - g**2) * (2 * cos**2 - 1))
        except TypeError:
            message = f"Invalid offset value(s): {g = }, {p = }, both must be {float}."
            raise ValueError(message) from None
        else:
            matrix = ((1 - g) * cos, (1 + g) * sin, (1 - g) * sin, (1 + g) * cos)
            return [coefficient * value for value in matrix]

    def _check_tuning(self, element_name, key):
        lo_freq = meta_dict[f"{element_name[0]}_LO"]["LO_freq"]
        int_freq = meta_dict[f"IF_freq"]

        if key == "LO":
            center = lo_freq
        elif key == "SB":
            center = lo_freq - int_freq

        span, rbw, ref_pow = self.span, self.rbw, self.ref_power

        self._sa.center = center
        self._sa.span = span
        self._sa.rbw = rbw
        self._sa.power = ref_pow

        fs, amps = self._sa.sweep()
        nominal_idx = math.ceil(self._sa.sweep_length / 2 + 1)
        # Search for the actual peak in the middle half of the sweep
        window = max(1, len(amps) // 4)
        left = max(0, nominal_idx - window)
        right = min(len(amps), nominal_idx + window)
        center_idx = left + int(np.argmax(amps[left:right]))
        stop, start = int(nominal_idx / 2), int(nominal_idx + (nominal_idx / 2))
        floor = (np.average(amps[:stop]) + np.average(amps[start:])) / 2
        contrast = amps[center_idx] - floor
        is_tuned = contrast < self.threshold
        real_center = fs[center_idx]
        self._initial_contrast = contrast
        print(
            f"[{key}] LO={lo_freq / 1e9:.6f} GHz  IF={int_freq / 1e6:.3f} MHz  "
            f"target={center / 1e9:.6f} GHz  SA peak={real_center / 1e9:.6f} GHz  "
            f"contrast={contrast:.2f} dBm  floor={floor:.2f} dBm"
        )
        return is_tuned, center_idx, floor

    def _get_contrast(self, center_idx: int, floor: float) -> float:
        """ """
        _, amps = self._sa.sweep()
        return abs(amps[center_idx] - floor)

    def _minimize_nm(
        self,
        fn,
        bounds,
    ) -> tuple[float]:
        """ """
        start_time = time.perf_counter()
        opt = {
            "fatol": self.threshold,
            "initial_simplex": self.simplex,
            "maxiter": self.maxiter,
        }
        result = scipy.optimize.minimize(
            fn, [0, 0], method="Nelder-Mead", bounds=bounds, options=opt
        )
        if result.success:
            time_, contrast = time.perf_counter() - start_time, fn(result.x)
            print(f"Minimized in {time_:.5}s with final {contrast = :.5}")
            if contrast > self.threshold:
                diff = contrast - self.threshold
                print(f"WARNING: Final contrast exceeds threshold by {diff}dBm")
            return result.x
        else:
            print(f"ERROR: Minimization unsuccessful, details: {result.message}")


if __name__ == "__main__":
    QUBIT = "q1"
    ELEMENT_NAME = "q1"
    OP_NAME = "X90"

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)

    LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]

    # set_LO_parameters(LOs)

    qmm = QuantumMachinesManager(host="192.168.0.134")    #OXFORD
    # qmm = QuantumMachinesManager(host="192.168.0.188")    #EBAY  
    QM = qmm.open_qm(config)
    qmm.clear_all_job_results()

    #spectrum_analyzer = SA124("SA", id=23276344)
    spectrum_analyzer = SA124("SA", id=23276284)
    spectrum_analyzer.connect()

    mxrtnr = MixerTuner(spectrum_analyzer)

    i_offset, q_offset = mxrtnr.tune_lo(element_name=ELEMENT_NAME)
    g_offset, p_offset = mxrtnr.tune_sb(element_name=ELEMENT_NAME)

    spectrum_analyzer.disconnect()

    print("\nPaste into make_meta_dict:")
    print(f'        "mixer_off_I": {i_offset},')
    print(f'        "mixer_off_Q": {q_offset},')
    print(f'        "mixer_gain": {g_offset},')
    print(f'        "mixer_phase": {p_offset},')
