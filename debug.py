from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

output_path = ROOT / "m_RB_1Q_qua_debug.py"

# Same qubit/config setup as the __main__ block in m_RB_1Q.py, so the generated
# elements/pulses referenced by make_prog() actually exist in `config`.
QUBIT = "q6"
CIRCUIT_DEPTHS = [4]
NUM_CIRCUITS = 1
SHOTS = 1

try:
    from qm import generate_qua_script
    from LabCode.Experiments.RB.m_RB_1Q import RB1Q
    from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
    from LabCode.Config.config_helpers import meta_dict_to_config_1Q

    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {"threshold": 0, "ground_threshold": 0}

    # Bypass __init__ (which would try to open a real QM connection) and just
    # populate what make_prog() needs to build the QUA program.
    rb = RB1Q.__new__(RB1Q)
    rb.meta_dict = meta_dict
    rb.config = config
    rb.element = QUBIT
    rb.circuit_depths = CIRCUIT_DEPTHS
    rb.max_depth = max(CIRCUIT_DEPTHS)
    rb.num_circuits = NUM_CIRCUITS
    rb.shots = SHOTS
    rb.calib_params = calib_params
    rb.calib_params_3state = None
    rb.use_3state = False
    rb.seed = 821467

    prog = rb.make_prog()
    source = generate_qua_script(prog, config)
    output_path.write_text(source, encoding="utf-8")
    print(f"Wrote QUA debug source to {output_path}")
except Exception as exc:
    output_path.write_text(
        "# QUA generation failed\n"
        f"# {exc}\n"
        "# Install the Quantum Machines SDK and required dependencies to generate the full script.\n",
        encoding="utf-8",
    )
    print(f"Could not generate QUA source: {exc}")
    print(f"Wrote fallback debug file to {output_path}")
