import numpy as np
import scipy as p
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import ODR, Model, RealData
import itertools

from scipy.stats import norm, multivariate_normal


# Bayesain Razor
def solve_poly(a, x):
    """
    y(x) = a₀ xᴺ + a₁ xᴺ⁻¹ + ... + a_N
    
    return: y, dy/da, d2y/dada  
    """

    N = len(a)
    powers = N - 1 - np.arange(N)           # from N-1 to 0
    dyda = x ** powers                      
    y = np.dot(a, dyda)
    d2 = np.zeros((N, N))                   # d2y/dada = 0
    return y, dyda[:, None], d2


def bayes_poly_map(selected_powers, mean_prior, inv_prior, likeli_inv_Cov, x, y):

    Phi = np.vstack([x ** p for p in selected_powers]).T

    inv_post = inv_prior + likeli_inv_Cov * (Phi.T @ Phi)
    Cov_post = np.linalg.inv(inv_post)
    a_map = Cov_post @ (inv_prior @ mean_prior + likeli_inv_Cov * Phi.T @ y)

    C = (1.0 / likeli_inv_Cov) * np.eye(len(x)) + Phi @ np.linalg.inv(inv_prior) @ Phi.T
    sign, logdetC = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("Covariance matrix C not positive definite")

    log_evidence = (-0.5 * len(x) * np.log(2.0 * np.pi)
                    -0.5 * logdetC
                    -0.5 * y @ np.linalg.solve(C, y))
    
    return a_map, log_evidence