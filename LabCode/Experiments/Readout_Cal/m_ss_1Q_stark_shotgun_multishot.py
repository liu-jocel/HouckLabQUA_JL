from pathlib import Path

from qm.qua import *
import numpy as np
import time
import h5py
from LabCode.CoreLib.experiment import ExperimentClass, qmm
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def _measure_ts(meta_dict, I, Q, timestamp_stream):
    """Demodulate resonator and capture a hardware timestamp."""
    measure(
        'read', meta_dict['r_name'], None,
        demod.full('integ_W1_' + meta_dict['r_name'], I, 'out'),
        demod.full('integ_W2_' + meta_dict['r_name'], Q, 'out'),
        timestamp_stream=timestamp_stream,
    )
    return I, Q


class ShotgunStarkSSMultishot(ExperimentClass):
    """
    Shotgun-sampled single-shot readout with simultaneous AC Stark shift (g/e only).

    Python drives the outer loop over (amp, sign, state) combos.  For each combo,
    a dedicated finite QUA program takes `numshots` single-shot measurements, then
    Python saves the result before executing the next combo.

    One complete pass through all shuffled combos = one cycle.
    The combo order is re-randomised every cycle.

    Config requirement
    ------------------
    Build the config with  MD["cw_amp"] = max(stark_amp_vec)  so that
    amp(a / amp_max) maps correctly to the physical amplitude a.

    Per-(amp, sign) HDF5 layout  (<stem>/<stem>_a{amp}_{pos|neg}.h5)
    -----------------------------------------------------------------
    I_g          (cycles, numshots)  float32   ground state prep
    Q_g          (cycles, numshots)  float32
    timestamps_g (cycles, numshots)  int64
    idx_g        (cycles,)           int64    1-based cycle counter for g rows
    I_e          (cycles, numshots)  float32   excited state prep
    Q_e          (cycles, numshots)  float32
    timestamps_e (cycles, numshots)  int64
    idx_e        (cycles,)           int64    1-based cycle counter for e rows
    amp          (1,)                float64
    sign         (1,)                int8
    Attributes: qubit, stark_detuning, amp_max, numshots
    """

    def __init__(self, element, stark_amp_vec, stark_detuning, runtime_seconds,
                 numshots=2000,
                 outer_folder='', path='ShotgunStarkSSMultishot', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)
        self.element         = element
        self.stark_amp_vec   = np.asarray(stark_amp_vec, dtype=float)
        self.amp_max         = float(np.max(np.abs(self.stark_amp_vec)))
        self.stark_detuning  = int(stark_detuning)
        self.runtime_seconds = runtime_seconds
        self.numshots        = int(numshots)

        # All unique (amp, sign, state) combinations — order re-randomised each cycle
        self.unique_combos = [
            (float(a), int(s), int(st))
            for a  in self.stark_amp_vec
            for s  in (1, -1)
            for st in (0, 1)
        ]

        self._run_and_save()

    # ── Per-combo QUA program ────────────────────────────────────────────────

    def _make_combo_prog(self, a, sgn, st):
        """
        Finite QUA program for a single (amp, sign, state) combo.

        Runs exactly `numshots` back-to-back single-shot measurements with
        (a, sgn, st) baked in as Python constants — no QUA branching overhead.
        """
        qubit      = self.meta_dict['q_name']
        resonator  = self.meta_dict['r_name']
        IF_freq    = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        stark_freq = self.meta_dict['IF_freq'] + (self.stark_detuning if sgn == 1
                                                   else -self.stark_detuning)
        reset_time = self.meta_dict['reset_time']
        read_len   = self.meta_dict['read_len']
        numshots   = self.numshots
        a_norm     = float(a / self.amp_max) if self.amp_max > 0 else 0.0

        with program() as prog:
            shot = declare(int)
            I    = declare(fixed)
            Q    = declare(fixed)

            I_stream  = declare_stream()
            Q_stream  = declare_stream()
            ts_stream = declare_stream()

            update_frequency(qubit, IF_freq)

            with for_(shot, 0, shot < numshots, shot + 1):
                if st == 1:                     # Python constant — baked in at compile time
                    play('X180', qubit)
                align(qubit, resonator)
                update_frequency(qubit, stark_freq)
                play("constant_tone" * amp(a_norm), qubit, duration=read_len * u.ns)
                _measure_ts(self.meta_dict, I, Q, ts_stream)
                align(qubit, resonator)
                update_frequency(qubit, IF_freq)
                save(I, I_stream)
                save(Q, Q_stream)
                wait(reset_time * u.ns, qubit)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
                ts_stream.save_all("timestamps")

        return prog

    # ── Per-(amp, sign) HDF5 files ───────────────────────────────────────────

    def _create_combo_h5(self):
        """
        Create one HDF5 file per unique (amp, sign) pair in a subdirectory.
        Each file holds g and e state data in separate _g / _e datasets.

        Returns
        -------
        combo_fnames : dict  (amp, sign) -> Path
        """
        stem   = Path(self.fname).stem
        subdir = Path(self.fname).parent / stem
        subdir.mkdir(exist_ok=True)

        combo_fnames = {}
        seen = set()
        for a, sgn, _ in self.unique_combos:
            key = (a, sgn)
            if key in seen:
                continue
            seen.add(key)

            sgn_str = "pos" if sgn == 1 else "neg"
            fpath   = subdir / f"{stem}_a{a:.3f}_{sgn_str}.h5"
            combo_fnames[key] = fpath

            with h5py.File(fpath, "w") as f:
                for suffix in ("g", "e"):
                    f.create_dataset(f"I_{suffix}",          shape=(0, self.numshots),
                                     maxshape=(None, self.numshots), dtype=np.float32)
                    f.create_dataset(f"Q_{suffix}",          shape=(0, self.numshots),
                                     maxshape=(None, self.numshots), dtype=np.float32)
                    f.create_dataset(f"timestamps_{suffix}", shape=(0, self.numshots),
                                     maxshape=(None, self.numshots), dtype=np.int64)
                    f.create_dataset(f"idx_{suffix}",        shape=(0,), maxshape=(None,), dtype=np.int64)
                f.create_dataset("amp",  data=np.array([a],   dtype=np.float64))
                f.create_dataset("sign", data=np.array([sgn], dtype=np.int8))
                f.attrs["qubit"]          = self.element
                f.attrs["stark_detuning"] = self.stark_detuning
                f.attrs["amp_max"]        = self.amp_max
                f.attrs["numshots"]       = self.numshots

        return combo_fnames

    # ── Live data collection ─────────────────────────────────────────────────

    def _run_and_save(self):
        combo_fnames = self._create_combo_h5()

        # Compile one program per unique combo up front
        combo_progs = {key: self._make_combo_prog(*key) for key in self.unique_combos}

        # Open QM once for all sequential executions
        qm = qmm.open_qm(self.config)

        t_start = time.time()
        cycle   = 0
        rng     = np.random.default_rng()

        while time.time() - t_start < self.runtime_seconds:
            cycle += 1

            # Re-randomise combo order each cycle
            order = self.unique_combos.copy()
            rng.shuffle(order)

            for a, sgn, st in order:
                job     = qm.execute(combo_progs[(a, sgn, st)])
                handles = job.result_handles
                handles.wait_for_all_values()

                I_vals  = handles.get("I").fetch_all()['value']
                Q_vals  = handles.get("Q").fetch_all()['value']
                ts_vals = handles.get("timestamps").fetch_all()['value']

                sfx   = "g" if st == 0 else "e"
                fpath = combo_fnames[(a, sgn)]
                with h5py.File(fpath, "a") as f:
                    for ds_key, vals in [(f"I_{sfx}",          I_vals),
                                         (f"Q_{sfx}",          Q_vals),
                                         (f"timestamps_{sfx}", ts_vals)]:
                        ds = f[ds_key]
                        n  = ds.shape[0]
                        ds.resize(n + 1, axis=0)
                        ds[n] = vals
                        ds.flush()
                    ds = f[f"idx_{sfx}"]
                    n  = ds.shape[0]
                    ds.resize(n + 1, axis=0)
                    ds[n] = cycle
                    ds.flush()

            elapsed = time.time() - t_start
            print(f"  cycle {cycle} complete  elapsed: {elapsed:.1f} s")

        subdir = Path(self.fname).parent / Path(self.fname).stem
        print(f"\nDone — {cycle} cycles saved → {subdir}/")

    # ── Simulation ──────────────────────────────────────────────────────────

    def simulate(self, a=None, sgn=1, st=0, duration=None):
        """Simulate a single combo's program and display the waveform report."""
        if a is None:
            a = float(self.stark_amp_vec[0])
        prog = self._make_combo_prog(a, sgn, st)
        if duration is None:
            cycles_per_shot = (self.meta_dict['read_len'] // 4
                               + self.meta_dict['reset_time'] // 4 + 200)
            duration = self.numshots * cycles_per_shot + 500
        return super().simulate(prog, self.config, duration=duration)


if __name__ == "__main__":
    import copy
    import matplotlib
    matplotlib.use("QtAgg")
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    # ── Configuration ────────────────────────────────────────────────────────
    SIMULATE = False

    QUBIT            = 'q1'
    STARK_AMP_VEC    = np.linspace(0, 0.2, 3)
    STARK_DETUNING   = 50 * u.MHz
    RUNTIME_SECONDS  = 48*60 * 60
    NUMSHOTS         = 10000

    # ── Build meta dict and config ───────────────────────────────────────────
    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    MD = copy.deepcopy(meta_dict)
    MD["cw_amp"] = max(abs(a) for a in STARK_AMP_VEC)
    MD["cw_len"] = meta_dict["read_len"]
    config = meta_dict_to_config_1Q(MD)

    if SIMULATE:
        exp = ShotgunStarkSSMultishot.__new__(ShotgunStarkSSMultishot)
        MD_sim = {**MD, "reset_time": 16 * u.ns}
        exp.meta_dict      = MD_sim
        exp.config         = config
        exp.element        = QUBIT
        exp.stark_detuning = int(STARK_DETUNING)
        exp.numshots       = NUMSHOTS
        exp.stark_amp_vec  = np.asarray(STARK_AMP_VEC)
        exp.amp_max        = float(np.max(np.abs(STARK_AMP_VEC)))
        exp.unique_combos  = [
            (float(a), int(s), int(st))
            for a  in STARK_AMP_VEC
            for s  in (1, -1)
            for st in (0, 1)
        ]
        exp.simulate(a=float(STARK_AMP_VEC[1]), sgn=1, st=0)

    else:
        exp = ShotgunStarkSSMultishot(
            QUBIT,
            STARK_AMP_VEC,
            STARK_DETUNING,
            RUNTIME_SECONDS,
            numshots=NUMSHOTS,
            outer_folder=meta_dict["global_params"]["outer_folder"],
            path=f"{QUBIT}",
            meta_dict=MD,
            suffix="stark_ss_multi",
            config=config,
        )
