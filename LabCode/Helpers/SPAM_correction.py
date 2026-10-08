import numpy as np
from scipy.optimize import minimize, LinearConstraint


def frobenius_norm(x, T, p_raw):
    p_corr = np.matrix([[x[i]] for i in range(len(x))])
    # p_corr = np.matrix([[x[0]],
    #                     [x[1]],
    #                     [x[2]],
    #                     [x[3]]])

    # print(np.linalg.norm(T @ p_corr - p_raw) ** 2)

    return np.linalg.norm(T @ p_corr - p_raw) ** 2


def SPAM_correct(T, p_raw):

    # SPAM correction through minimization of Frobenius norm. dimension of state space is 4 by default (2 qubit 2 level)

    x = np.zeros(len(p_raw))

    # define the constraints
    lb = np.zeros(len(x))
    ub = np.ones(len(x))
    linear_constraint = LinearConstraint(np.ones(len(x)), [1], [1])

    # define the initial guess for x
    x0 = np.ones(len(x)) / len(x)

    # solve the optimization problem
    result = minimize(frobenius_norm, x0, args=(T, p_raw), method='SLSQP', bounds=list(zip(lb, ub)),
                      constraints=[linear_constraint])

    return result.x

def correct_counts_array(T,counts_array):

    counts_array_cor = np.zeros(np.shape(counts_array))

    num_exp = np.shape(counts_array)[1]

    for i in range(num_exp):
        p_raw = np.matrix(counts_array[:,i]/np.sum(counts_array[:,i])).transpose()

        p_cor = SPAM_correct(T,p_raw)

        counts_array_cor[:,i] = p_cor*np.sum(counts_array[:,i])

    return counts_array_cor