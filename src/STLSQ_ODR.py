import numpy as np
import scipy as p
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy import odr
from scipy.stats import t
from scipy.stats import chi2

# STLSQ_ODR
# L2([-1,1]) normalization factor: factor[k] = sqrt((2k+1)/2)
def l2_norm_factors(deg: int) -> np.ndarray:
    return np.sqrt((2 * np.arange(deg + 1) + 1) / 2)

def _odr_fit_powers(x: np.ndarray,
                    y: np.ndarray,
                    powers: np.ndarray,
                    factors: np.ndarray,
                    beta0: np.ndarray | None = None,
                    sx: float | np.ndarray = 0.06,
                    sy: float | np.ndarray = 0.1) -> np.ndarray:
    powers = np.asarray(powers)

    def poly_model(beta, x):
        out = np.zeros_like(x, dtype=float)
        for j, p in enumerate(powers):
            out += beta[j] * (x**p) * factors[p]  # normalized basis
        return out

    model = odr.Model(poly_model)
    data  = odr.RealData(x, y, sx=sx, sy=sy)

    if beta0 is None:
        # Use the normalized Θ submatrix to perform LSQ as the initial guess
        Theta_norm = np.vstack([(x**p) * factors[p] for p in powers]).T
        beta0 = np.linalg.lstsq(Theta_norm, y, rcond=None)[0].ravel()

    out = odr.ODR(data, model, beta0=beta0).run()
    return out.beta  # coefficients under the normalized basis (corresponding to powers)

# STLSQ + ODR (internally uses normalized coefficients, finally denormalizes)
def STLSQ_ODR(x: np.ndarray,
              dXdt: np.ndarray,
              deg: int,
              lambd: float,
              n_iter: int = 10,
              sx: float = 0.06,
              sy: float = 0.1) -> np.ndarray:
    """
    Returns: beta (deg+1, n_states) — denormalized, can be directly multiplied with x^k
    """
    factors = l2_norm_factors(deg)
    n_features, n_states = deg + 1, dXdt.shape[1]

    # Xi_norm: coefficients under the normalized basis
    Xi_norm = np.zeros((n_features, n_states))
    all_powers = np.arange(deg + 1)

    # Initial fitting (all powers)
    for k in range(n_states):
        Xi_norm[:, k] = _odr_fit_powers(x, dXdt[:, k], all_powers, factors, sx=sx, sy=sy)

    # Iterative sparsification (thresholding on the normalized scale)
    for _ in range(n_iter):
        small = np.abs(Xi_norm) < lambd
        Xi_norm[small] = 0.0

        for k in range(n_states):
            powers_kept = np.flatnonzero(~small[:, k])
            if powers_kept.size == 0:
                continue
            # Refit on the retained powers
            beta_sub = _odr_fit_powers(x, dXdt[:, k], powers_kept, factors, sx=sx, sy=sy)
            # Write back to the corresponding positions (still under normalized basis)
            Xi_norm[powers_kept, k] = beta_sub
            # Deleted terms remain as 0

    # Denormalize to standard polynomial coefficients: beta = Xi_norm * factor
    beta = Xi_norm * factors[:, None]
    return beta