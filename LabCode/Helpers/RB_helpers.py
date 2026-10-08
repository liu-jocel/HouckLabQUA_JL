"""
Created on Mar 19 2024

@author: hpreste
"""

import numpy as np
import cirq
from scipy.optimize import minimize
from LabCode.Helpers.gate_macros import UGate, phased_XZ

def fit_leakage_RB(d, L1, L2, pL):
    """
    Fit function for the leakage population of an RB experiment.
    Taken from:
    Phys. Rev. A 97, 032306 – Published 8 March, 2018
    DOI: https://doi.org/10.1103/PhysRevA.97.032306

    **ARGS:
    d: circiut_depth,
    L1: Leakage Rate
    L2: Seepage Rate
    pL: initial population in the leakage subspace
    """
    A = L2/(L1+L2)
    B = L1/(L1+L2)
    return A - (B - pL)*(1-L1-L2)**d

def fit_L1_L2(d, L1, L2, e_spam):
    """
    Leakage RB model for |2> state population. Wood & Gambetta (2018).
    L1: leakage rate, L2: seepage rate, e_spam: initial |2> population (SPAM).
    Steady state: L1/(L1+L2).
    """
    A = L1 / (L1 + L2)
    B = L1 / (L1 + L2) - e_spam
    return A - B * (1 - L1 - L2)**d

def get_1q_clif_cirq(q):
    circuits = [[],
                [cirq.X(q)],
                [cirq.Y(q)],
                [cirq.Y(q), cirq.X(q)],
                [cirq.X(q) ** 0.5, cirq.Y(q) ** 0.5],
                [cirq.X(q) ** 0.5, cirq.Y(q) ** -0.5],
                [cirq.X(q) ** -0.5, cirq.Y(q) ** 0.5],
                [cirq.X(q) ** -0.5, cirq.Y(q) ** -0.5],
                [cirq.Y(q) ** 0.5, cirq.X(q) ** 0.5],
                [cirq.Y(q) ** 0.5, cirq.X(q) ** -0.5],
                [cirq.Y(q) ** -0.5, cirq.X(q) ** 0.5],
                [cirq.Y(q) ** -0.5, cirq.X(q) ** -0.5],
                [cirq.X(q) ** 0.5],
                [cirq.X(q) ** -0.5],
                [cirq.Y(q) ** 0.5],
                [cirq.Y(q) ** -0.5],
                [cirq.X(q) ** -0.5, cirq.Y(q) ** 0.5, cirq.X(q) ** 0.5],
                [cirq.X(q) ** -0.5, cirq.Y(q) ** -0.5, cirq.X(q) ** 0.5],
                [cirq.X(q), cirq.Y(q) ** 0.5],
                [cirq.X(q), cirq.Y(q) ** -0.5],
                [cirq.Y(q), cirq.X(q) ** 0.5],
                [cirq.Y(q), cirq.X(q) ** -0.5],
                [cirq.X(q) ** 0.5, cirq.Y(q) ** 0.5, cirq.X(q) ** 0.5],
                [cirq.X(q) ** -0.5, cirq.Y(q) ** 0.5, cirq.X(q) ** -0.5]
                ]

    names = [['I'],
             ['x'],
             ['y'],
             ['y', 'x'],
             ['x2', 'y2'],
             ['x2', 'my2'],
             ['mx2', 'y2'],
             ['mx2', 'my2'],
             ['y2', 'x2'],
             ['y2', 'mx2'],
             ['my2', 'x2'],
             ['my2', 'mx2'],
             ['x2'],
             ['mx2'],
             ['y2'],
             ['my2'],
             ['mx2', 'y2', 'x2'],
             ['mx2', 'my2', 'x2'],
             ['x', 'y2'],
             ['x', 'my2'],
             ['y', 'x2'],
             ['y', 'mx2'],
             ['x2', 'y2', 'x2'],
             ['mx2', 'y2', 'mx2']
             ]
    return circuits, names


def UGate_matrix(theta, phi, lam):
    """
    Implements a UGate matrix model,
    all parameters are bounded [0,1)
    """
    matrix = [[np.cos(np.pi * theta), -1j*np.exp(2 * np.pi * 1j * lam) * np.sin(np.pi * theta)],
              [-1j*np.exp(2 * np.pi * 1j * phi) * np.sin(np.pi * theta),
               np.exp(2 * np.pi * 1j * (phi + lam)) * np.cos(np.pi * theta)]]
    return np.array(matrix)

def PMW2_matrix(sigma, theta, omega):
    """
    Implements PMW2 arbitrary Unitary: X_sigma(theta).X_pi(omega)
    https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.L022031
    """
    gamma = (sigma+1)/2
    beta = 3/2-omega
    alpha = theta - omega
    matrix = [[np.exp(2j*np.pi*alpha)*np.cos(np.pi*gamma), -np.exp(-2j*np.pi*beta)*np.sin(np.pi*gamma)],
              [np.exp(2j*np.pi*beta)*np.sin(np.pi*gamma), np.exp(2j*np.pi*alpha)*np.cos(np.pi*gamma)]]
    return np.array(matrix)
def fit_unitary_to_gate_arb(unitary, gate, initial_guess):
    """
    Regression for arbitrary unitary
    """

    def cost_function(params, target_matrix):
        return np.sum(np.abs(gate(*params) - target_matrix) ** 2)

    result = minimize(cost_function, initial_guess, args=(unitary,), method='Nelder-Mead')
    gate_params = result.x % 1  # Normalize the Guess frm [0, 1)
    return gate_params


def generate_rb_phasedXZ(m, a, x, z):
    """
    Function that generates a randomized benchmarking sequence of length m and computed the inverse operations for the resulting circuits.

    m : int, length of RB cliffords

    a, x, z, : floats in [0,1), exponents in cirq PhasedXZ 1Q gate

    """

    reference = cirq.Circuit()
    interleaved = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)

    RB_seq = np.random.randint(0, high=24, size=m)
    phxz = cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
    circuits, names = get_1q_clif_cirq(q1)

    for gate in RB_seq:
        reference += circuits[gate]
        interleaved += (circuits[gate] + [phxz(q1)])

    try:
        reference_inverse = reference ** -1
        reference_inverse_unitary = reference_inverse.unitary()
        reference_inverse_PhasedXZ = cirq.PhasedXZGate.from_matrix(reference_inverse_unitary)
        ref_theta_out, ref_phi_out, ref_lam_out = (reference_inverse_PhasedXZ.axis_phase_exponent,
                              reference_inverse_PhasedXZ.x_exponent, reference_inverse_PhasedXZ.z_exponent)
    except IndexError:
        ref_theta_out, ref_phi_out, ref_lam_out = [0, 0, 0]

    interleaved_inverse = interleaved ** -1
    interleaved_inverse_unitary = interleaved_inverse.unitary()
    interleaved_inverse_PhasedXZ = cirq.PhasedXZGate.from_matrix(interleaved_inverse_unitary)


    print("")
    print("##################################################################################")
    print("")

    print("Reference Circuit:")
    # print(reference)
    print("Reference Unitary:")
    print(reference.unitary())
    print("Reference Inverse Unitary:")
    print(reference_inverse_unitary)
    print("The following matrices should be close to identity")
    print(reference.unitary() @ reference_inverse_unitary)
    print(interleaved.unitary() @ interleaved_inverse_unitary)



    out_dict = {
        "gate": phased_XZ,
        "gate_args": (a, x, z,),
        "sequence": RB_seq,
        "reference_inverse": (ref_theta_out, ref_phi_out, ref_lam_out),
        "interleaved_inverse": (interleaved_inverse_PhasedXZ.axis_phase_exponent,
                                interleaved_inverse_PhasedXZ.x_exponent, interleaved_inverse_PhasedXZ.z_exponent,)
    }
    return out_dict


def generate_rb_UGate(m, theta, phi, lam,):
    #TODO: investegate inverting identity matrix
    """
    Function that generates a randomized benchmarking sequence of length m and computed the inverse operations for the resulting circuits.

    m : int, length of RB cliffords

    a, x, z, : floats in [0,1), exponents in cirq PhasedXZ 1Q gate

    """
    reference = cirq.Circuit()
    interleaved = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)

    RB_seq = np.random.randint(0, high=24, size=m)
    unitary_matrix = UGate_matrix(theta, phi, lam)

    unitary_phased_xz = cirq.PhasedXZGate.from_matrix(unitary_matrix)

    circuits, names = get_1q_clif_cirq(q1)

    for gate in RB_seq:
        reference += circuits[gate]
        interleaved += (circuits[gate] + [unitary_phased_xz(q1)])

    try:
        reference_inverse = reference ** -1
        reference_inverse_unitary = reference_inverse.unitary()
        ref_theta_out, ref_phi_out, ref_lam_out = fit_unitary_to_gate_arb(reference_inverse_unitary, UGate_matrix,
                                                                      [0.1, 0.1, 0.1])
    except IndexError:
        ref_theta_out, ref_phi_out, ref_lam_out = [0, 0, 0]

    interleaved_inverse = interleaved ** -1
    interleaved_inverse_unitary = interleaved_inverse.unitary()
    int_theta_out, int_phi_out, int_lam_out = fit_unitary_to_gate_arb(interleaved_inverse_unitary, UGate_matrix,
                                                                      [0.1, 0.1, 0.1])
    print("")
    print("##################################################################################")
    print("")
    print("The following matrices should be close to identity")
    print(reference.unitary() @ reference_inverse_unitary)
    print(interleaved.unitary() @ interleaved_inverse_unitary)

    out_dict = {
        "gate": UGate,
        "gate_args": (theta, phi, lam),
        "sequence": RB_seq,
        "reference_inverse": (ref_theta_out, ref_phi_out, ref_lam_out),
        "interleaved_inverse": (int_theta_out, int_phi_out, int_lam_out),
    }
    return out_dict

def invert_u_gate(theta, phi, lam):
    c = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)
    unitary_matrix = UGate_matrix(theta, phi, lam)

    unitary_phased_xz = cirq.PhasedXZGate.from_matrix(unitary_matrix)
    c += unitary_phased_xz(q1)
    c_inv = c**-1
    c_inv_unitary = c_inv.unitary()
    theta_out, phi_out, lam_out = fit_unitary_to_gate_arb(c_inv_unitary, UGate_matrix,
                                                                      [0.1, 0.1, 0.1])
    return theta_out, phi_out, lam_out

def invert_phxz(a, x, z):
    c = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)
    phxz = cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
    c += phxz(q1)
    c_inv = c**-1
    c_inv_unitary = c_inv.unitary()
    c_inv_phxz = cirq.PhasedXZGate.from_matrix(c_inv_unitary)
    return (c_inv_phxz.axis_phase_exponent, c_inv_phxz.x_exponent, c_inv_phxz.z_exponent)


def generate_rb_Unitary(m, gate_dict):
    #TODO: investegate inverting identity matrix
    """
    Function that generates a randomized benchmarking sequence of length m and computed the inverse operations for the resulting circuits.

    m : int, length of RB cliffords

    """
    gate_unitary = gate_dict["gate_unitary"]
    gate_func = gate_dict["gate_func"]
    theta, phi, lam = gate_dict["angles"]


    reference = cirq.Circuit()
    interleaved = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)

    RB_seq = np.random.randint(0, high=24, size=m)
    unitary_matrix = gate_unitary(theta, phi, lam)

    unitary_phased_xz = cirq.PhasedXZGate.from_matrix(unitary_matrix)

    circuits, names = get_1q_clif_cirq(q1)

    for gate in RB_seq:
        reference += circuits[gate]
        interleaved += (circuits[gate] + [unitary_phased_xz(q1)])

    try:
        reference_inverse = reference ** -1
        reference_inverse_unitary = reference_inverse.unitary()
        ref_theta_out, ref_phi_out, ref_lam_out = fit_unitary_to_gate_arb(reference_inverse_unitary, gate_unitary,
                                                                      [0.1, 0.1, 0.1])
    except IndexError:
        ref_theta_out, ref_phi_out, ref_lam_out = [0, 0, 0]

    interleaved_inverse = interleaved ** -1
    interleaved_inverse_unitary = interleaved_inverse.unitary()
    int_theta_out, int_phi_out, int_lam_out = fit_unitary_to_gate_arb(interleaved_inverse_unitary, gate_unitary,
                                                                      [0.1, 0.1, 0.1])
    print("")
    print("##################################################################################")
    print("")
    print("The following matrices should be close to identity")
    print(reference.unitary() @ reference_inverse_unitary)
    print(interleaved.unitary() @ interleaved_inverse_unitary)

    out_dict = {
        "gate": gate_func,
        "gate_args": (theta, phi, lam),
        "sequence": RB_seq,
        "reference_inverse": (ref_theta_out, ref_phi_out, ref_lam_out),
        "interleaved_inverse": (int_theta_out, int_phi_out, int_lam_out),
    }
    return out_dict

def invert_unitary(sigma, theta, omega, unitary_func):
    c = cirq.Circuit()
    q1, q2 = cirq.LineQubit.range(2)
    unitary_matrix = unitary_func(theta, phi, lam)

    unitary_phased_xz = cirq.PhasedXZGate.from_matrix(unitary_matrix)
    c += unitary_phased_xz(q1)
    c_inv = c**-1
    c_inv_unitary = c_inv.unitary()
    sigma_out, theta_out, omega_out = fit_unitary_to_gate_arb(c_inv_unitary, unitary_func,
                                                                      [0.1, 0.1, 0.1])
    return sigma_out, theta_out, omega_out

if __name__ == "__main__":
    x, z, a = np.random.rand(3)
    rb_PXZ = generate_rb_phasedXZ(1024, a, x, z)
    theta, phi, lam = np.random.rand(3)
    rb_UGate = generate_rb_UGate(1024, theta, phi, lam)
    sigma, theta, omega = np.random.rand(3)
    rb_PMW2 = generate_rb_Unitary(1024, PMW2_matrix, sigma, theta, omega)
    print("")
    print("##################################################################################")
    print("")
    print("RB for PhasedXZ")
    print(rb_PXZ)
    print("")
    print("##################################################################################")
    print("")
    print("RB for UGate")
    print(rb_UGate)
    print("")
    print("##################################################################################")
    print("")
    print("RB for PMW2 using arbitrary function:")
    print(rb_PMW2)