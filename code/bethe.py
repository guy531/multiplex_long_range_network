"""Closed-form (Bethe / tree) quantities for the 1D long-range model."""
import numpy as np
from scipy.special import zeta
from scipy.optimize import minimize_scalar, brentq

def logPplus(u, sigma, M=400):
    """log prod_{r>=1} (1 - u r^-(1+sigma)),  0 <= u < 1."""
    s = 1.0 + sigma
    m = np.arange(1, M + 1)
    return -np.sum(u ** m / m * zeta(m * s))

def q_active(u, sigma):
    """Probability that a node has at least one bond, at parameter u."""
    return -np.expm1(2.0 * logPplus(u, sigma))

def C_schulman(sigma):
    return 1.0 / (2.0 * zeta(1.0 + sigma))

def C_bethe_mux(sigma):
    """min_u u / q(u)^2 : threshold of the Bethe approximation to the closure, full coupling.
    Returns (C, u*, jump = q(u*)^2)."""
    f = lambda u: u / q_active(u, sigma) ** 2
    res = minimize_scalar(f, bounds=(1e-6, 0.999), method='bounded', options={'xatol': 1e-12})
    return res.fun, res.x, q_active(res.x, sigma) ** 2

def rho_bethe(lam, sigma):
    """Percolation probability of Schulman's Bethe lattice at parameter lam:
    largest root of 1 - rho = P_+(lam*rho)^2."""
    if 2 * lam * zeta(1 + sigma) <= 1:
        return 0.0
    f = lambda r: 1 - r - np.exp(2 * logPplus(lam * r, sigma))
    return brentq(f, 1e-12, 1.0)

if __name__ == "__main__":
    gori = {0.1: 0.047685, 0.2: 0.093211, 0.3: 0.140546, 0.4: 0.193471, 0.5: 0.25482,
            0.6: 0.327098, 0.7: 0.413752, 0.8: 0.521001, 0.9: 0.66408}
    print("sigma  C_S      C_c(Gori)  C_B^mux   ratio   u*      jump")
    for s in [0.02, 0.1, 0.2, 0.3, 1/3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        C, u, j = C_bethe_mux(s)
        print(f"{s:.3f}  {C_schulman(s):.5f}  {gori.get(round(s,2), float('nan')):.5f}   {C:.5f}  {C/C_schulman(s):.4f}  {u:.4f}  {j:.4f}")


def C_star(sigma):
    """Rigorous multiplex lower bound of the earlier note: root of 2 C zeta kappa(C) = 1,
    kappa = 1 - P_+(C) P_+(C/q), q = 1 - P_+(C)^2."""
    z = zeta(1.0 + sigma)

    def F(C):
        lp = logPplus(C, sigma)
        q = -np.expm1(2 * lp)
        return 2 * C * z * (-np.expm1(lp + logPplus(C / q, sigma))) - 1.0
    return brentq(F, 1e-6, 0.9)
