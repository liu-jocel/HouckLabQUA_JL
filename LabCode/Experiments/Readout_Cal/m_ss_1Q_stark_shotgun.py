from pathlib import Path

from qm.qua import *
import numpy as np
import time
import h5py
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qm_result_fetcher import QMResultFetcher
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

FETCH_INTERVAL_S = 5  # seconds between live-fetch polls


def _measure_ts(meta_dict, I, Q, timestamp_stream):
    """Demodulate resonator and capture a hardware timestamp."""
    measure(
        'read', meta_dict['r_name'], None,
        demod.full('integ_W1_' + meta_dict['r_name'], I, 'out'),
        demod.full('integ_W2_' + meta_dict['r_name'], Q, 'out'),
        timestamp_stream=timestamp_stream,
    )
    return I, Q


class ShotgunStarkSS(ExperimentClass):
    """
    Shotgun-sampled single-shot readout with simultaneous AC Stark shift (g/e only).

    All (stark_amp, detuning_sign, state_prep) combinations are built once, shuffled
    randomly in Python, then loaded into QUA as fixed arrays.  The OPX runs an
    infinite loop cycling through that shuffled sequence; the experiment runs until
    runtime_seconds elapses, then the job is halted.

    Pulse sequence per slot
    -----------------------
    1. Set qubit → IF_freq
    2. If state == 1: play X180  (|g> needs no pulse)
    3. align(qubit, resonator)
    4. Set qubit → IF_freq ± stark_detuning  (sign from slot)
    5. play "constant_tone" * amp(a_norm) on qubit  ─┐ simultaneous on
       measure resonator, capture timestamp          ─┘ independent timelines
    6. align, restore qubit → IF_freq
    7. wait resonator_dump_time

    Config requirement
    ------------------
    Build the config with  MD["cw_amp"] = max(stark_amp_vec)  so that the QUA
    amp() scale factor maps [0, 1] → [0, amp_max] in physical units.

    HDF5 layout
    -----------
    I, Q         (cycles, N)  float32   raw demodulated quadratures per cycle
    timestamps   (cycles, N)  int64     hardware timestamps per cycle
    seq_amp      (N,)         float64   Stark amplitude for each slot (fixed)
    seq_sign     (N,)         int8      +1 / -1 detuning sign for each slot (fixed)
    seq_state    (N,)         int8      0=g, 1=e state prep for each slot (fixed)
    Attributes: qubit, stark_detuning, stark_amp_vec, amp_max, runtime_seconds
    """

    def __init__(self, element, stark_amp_vec, stark_detuning, runtime_seconds,
                 outer_folder='', path='ShotgunStarkSS', suffix='data',
                 meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder, suffix=suffix,
                         meta_dict=meta_dict, config=config)
        self.element        = element
        self.stark_amp_vec  = np.asarray(stark_amp_vec, dtype=float)
        self.amp_max        = float(np.max(np.abs(self.stark_amp_vec)))
        self.stark_detuning = int(stark_detuning)
        self.runtime_seconds = runtime_seconds

        # Build and shuffle all (amp, sign, state) combinations once
        combos = [
            (a, s, st)
            for a  in self.stark_amp_vec
            for s  in (1, -1)
            for st in (0, 1)
        ]
        np.random.default_rng().shuffle(combos)

        self.seq_amp   = np.array([c[0] for c in combos], dtype=float)
        self.seq_sign  = np.array([c[1] for c in combos], dtype=np.int8)
        self.seq_state = np.array([c[2] for c in combos], dtype=np.int8)
        self.N = len(combos)

        # Normalised to [0, 1] for QUA amp(); config must set cw_amp = amp_max
        if self.amp_max > 0:
            self._seq_amp_norm = (self.seq_amp / self.amp_max).tolist()
        else:
            self._seq_amp_norm = [0.0] * self.N

        self.job = self.acquire(self.make_prog(), self.config)
        self._run_and_save()

    # ── QUA program ─────────────────────────────────────────────────────────

    def make_prog(self, simulate=False):
        qubit     = self.meta_dict['q_name']
        resonator = self.meta_dict['r_name']
        IF_freq   = self.meta_dict['IF_freq'] + self.meta_dict['detuning']
        pos_freq  = self.meta_dict['IF_freq'] + self.stark_detuning
        neg_freq  = self.meta_dict['IF_freq'] - self.stark_detuning
        reset_time = self.meta_dict["reset_time"]
        read_len  = self.meta_dict['read_len']
        N         = self.N

        with program() as prog:
            x   = declare(int)
            a   = declare(fixed)
            sgn = declare(int)
            st  = declare(int)
            I   = declare(fixed)
            Q   = declare(fixed)

            amp_list   = declare(fixed, size=N)
            sign_list  = declare(int,   size=N)
            state_list = declare(int,   size=N)

            I_stream  = declare_stream()
            Q_stream  = declare_stream()
            ts_stream = declare_stream()

            # Load shuffled sequence at compile time
            for k, v in enumerate(self._seq_amp_norm):
                assign(amp_list[k], v)
            for k, v in enumerate(self.seq_sign.tolist()):
                assign(sign_list[k], int(v))
            for k, v in enumerate(self.seq_state.tolist()):
                assign(state_list[k], int(v))

            def _one_cycle():
                with for_(x, 0, x < N, x + 1):
                    assign(a,   amp_list[x])
                    assign(sgn, sign_list[x])
                    assign(st,  state_list[x])

                    # State prep
                    update_frequency(qubit, IF_freq)
                    with if_(st == 1):
                        play('X180', qubit)

                    # Stark tone || readout — simultaneous on independent timelines
                    align(qubit, resonator)
                    with if_(sgn == 1):
                        update_frequency(qubit, pos_freq)
                    with else_():
                        update_frequency(qubit, neg_freq)
                    play("constant_tone" * amp(a), qubit, duration=read_len * u.ns)
                    _measure_ts(self.meta_dict, I, Q, ts_stream)
                    align(qubit, resonator)
                    update_frequency(qubit, IF_freq)

                    save(I, I_stream)
                    save(Q, Q_stream)

                    wait(reset_time * u.ns, qubit)

            if simulate:
                _one_cycle()          # single pass for waveform inspection
            else:
                with infinite_loop_():
                    _one_cycle()      # loops until job.halt()

            with stream_processing():
                # buffer(N) groups one complete cycle into one row
                I_stream.buffer(N).save_all("I")
                Q_stream.buffer(N).save_all("Q")
                ts_stream.buffer(N).save_all("timestamps")

        return prog

    # ── HDF5 setup ──────────────────────────────────────────────────────────

    def _create_h5(self):
        """Create HDF5 with resizable I/Q/timestamp datasets and fixed metadata."""
        with h5py.File(self.fname, "w") as f:
            # Resizable data — rows appended as cycles complete
            f.create_dataset("I",          shape=(0, self.N), maxshape=(None, self.N), dtype=np.float32)
            f.create_dataset("Q",          shape=(0, self.N), maxshape=(None, self.N), dtype=np.float32)
            f.create_dataset("timestamps", shape=(0, self.N), maxshape=(None, self.N), dtype=np.int64)
            f.create_dataset("idx",        shape=(0,),        maxshape=(None,),        dtype=np.int64)
            # Fixed metadata — written once
            f.create_dataset("seq_amp",   data=self.seq_amp)
            f.create_dataset("seq_sign",  data=self.seq_sign)
            f.create_dataset("seq_state", data=self.seq_state)
            f.attrs["qubit"]           = self.element
            f.attrs["stark_detuning"]  = self.stark_detuning
            f.attrs["stark_amp_vec"]   = self.stark_amp_vec
            f.attrs["amp_max"]         = self.amp_max
            f.attrs["runtime_seconds"] = self.runtime_seconds

    # ── Per-combo HDF5 files ─────────────────────────────────────────────────

    def _create_combo_h5(self):
        """
        Create one HDF5 file per unique (amp, sign, state) combo.

        Files are placed in a subdirectory named after the main file's stem,
        e.g.  <stem>/  <stem>_a0.100_pos_st0.h5

        Returns
        -------
        combo_to_col : dict  (amp, sign, state) -> column index in (cycles, N) array
        combo_fnames : dict  (amp, sign, state) -> Path
        """
        stem   = Path(self.fname).stem
        subdir = Path(self.fname).parent / stem
        subdir.mkdir(exist_ok=True)

        combo_to_col = {}
        combo_fnames = {}

        for k in range(self.N):
            a   = float(self.seq_amp[k])
            sgn = int(self.seq_sign[k])
            st  = int(self.seq_state[k])
            key = (a, sgn, st)
            if key in combo_to_col:
                continue

            combo_to_col[key] = k
            sgn_str = "pos" if sgn == 1 else "neg"
            fpath   = subdir / f"{stem}_a{a:.3f}_{sgn_str}_st{st}.h5"
            combo_fnames[key] = fpath

            with h5py.File(fpath, "w") as f:
                f.create_dataset("I",          shape=(0,), maxshape=(None,), dtype=np.float32)
                f.create_dataset("Q",          shape=(0,), maxshape=(None,), dtype=np.float32)
                f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.int64)
                f.create_dataset("idx",        shape=(0,), maxshape=(None,), dtype=np.int64)
                f.attrs["amp"]            = a
                f.attrs["sign"]           = sgn
                f.attrs["state"]          = st
                f.attrs["qubit"]          = self.element
                f.attrs["stark_detuning"] = self.stark_detuning
                f.attrs["amp_max"]        = self.amp_max

        return combo_to_col, combo_fnames

    # ── Live data collection ─────────────────────────────────────────────────

    def _run_and_save(self):
        self._create_h5()
        combo_to_col, combo_fnames = self._create_combo_h5()

        fetcher   = QMResultFetcher(self.job.result_handles)
        t_start   = time.time()
        new_count = 0

        while not fetcher.is_done_fetching:
            elapsed = time.time() - t_start
            if elapsed > self.runtime_seconds:
                self.job.halt()

            new_data = fetcher.fetch()
            last_count, new_count = fetcher.counts
            print(f"  cycles: {new_count}  elapsed: {elapsed:.1f} s", end="\r")

            if new_data:
                # 1-based cycle indices for this batch
                cycle_idx = np.arange(last_count + 1, new_count + 1, dtype=np.int64)

                # Main aggregated file
                with h5py.File(self.fname, "a") as f:
                    for key in ("I", "Q", "timestamps"):
                        ds = f[key]
                        ds.resize(new_count, axis=0)
                        ds[last_count:new_count] = new_data[key]
                        ds.flush()
                    ds = f["idx"]
                    ds.resize(new_count, axis=0)
                    ds[last_count:new_count] = cycle_idx
                    ds.flush()

                # Per-combo files: extract one column per (amp, sign, state) slot
                for (a, sgn, st), col in combo_to_col.items():
                    fpath = combo_fnames[(a, sgn, st)]
                    with h5py.File(fpath, "a") as f:
                        for key in ("I", "Q", "timestamps"):
                            ds       = f[key]
                            new_vals = new_data[key][:, col]
                            n        = ds.shape[0]
                            ds.resize(n + len(new_vals), axis=0)
                            ds[n:]   = new_vals
                            ds.flush()
                        ds = f["idx"]
                        n  = ds.shape[0]
                        ds.resize(n + len(cycle_idx), axis=0)
                        ds[n:] = cycle_idx
                        ds.flush()

            time.sleep(FETCH_INTERVAL_S)

        print(f"\nDone — {new_count} cycles saved → {self.fname}")

    # ── Simulation ──────────────────────────────────────────────────────────

    def simulate(self, duration=None):
        """Simulate one cycle and display the waveform report.

        duration : int, optional
            Clock cycles (4 ns each). Auto-computed from read_len and N if omitted.
        """
        if duration is None:
            cycles_per_slot = self.meta_dict['read_len'] // 4 + 200
            duration = self.N * cycles_per_slot + 500
        return super().simulate(self.make_prog(simulate=True), self.config, duration=duration)


if __name__ == "__main__":
    import copy
    import matplotlib
    matplotlib.use("QtAgg")
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    # ── Configuration ────────────────────────────────────────────────────────
    SIMULATE = False           # False → run on real hardware

    QUBIT            = 'q1'
    STARK_AMP_VEC    = np.linspace(0, 0.4, 50)
    STARK_DETUNING   = 50 * u.MHz
    RUNTIME_SECONDS  = 60*60    # ignored when SIMULATE = True

    # ── Build meta dict and config ───────────────────────────────────────────
    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    MD = copy.deepcopy(meta_dict)
    MD["cw_amp"] = max(abs(a) for a in STARK_AMP_VEC)
    MD["cw_len"] = meta_dict["read_len"]
    config = meta_dict_to_config_1Q(MD)

    if SIMULATE:
        # Bypass __init__ — set only what make_prog needs.
        # Override resonator_dump_time so all slots fit in the simulation window.
        exp = ShotgunStarkSS.__new__(ShotgunStarkSS)
        MD_sim = {**MD, "reset_time": 16 * u.ns}
        exp.meta_dict      = MD_sim
        exp.config         = config
        exp.element        = QUBIT
        exp.stark_detuning = int(STARK_DETUNING)

        amp_max = float(max(abs(a) for a in STARK_AMP_VEC)) if any(STARK_AMP_VEC) else 1.0
        combos  = [(a, s, st) for a in STARK_AMP_VEC for s in (1, -1) for st in (0, 1)]
        np.random.default_rng().shuffle(combos)
        exp.seq_amp   = np.array([c[0] for c in combos], dtype=float)
        exp.seq_sign  = np.array([c[1] for c in combos], dtype=np.int8)
        exp.seq_state = np.array([c[2] for c in combos], dtype=np.int8)
        exp.N         = len(combos)
        exp._seq_amp_norm = (exp.seq_amp / amp_max).tolist() if amp_max > 0 else [0.0] * exp.N

        exp.simulate()   # duration auto-computed from read_len and N

    else:
        exp = ShotgunStarkSS(
            QUBIT,
            STARK_AMP_VEC,
            STARK_DETUNING,
            RUNTIME_SECONDS,
            outer_folder=meta_dict["global_params"]["outer_folder"],
            path=f"{QUBIT}",
            meta_dict=MD,
            suffix="stark_ss",
            config=config,
        )
