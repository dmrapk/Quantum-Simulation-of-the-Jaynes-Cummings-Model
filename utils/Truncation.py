import math
import numpy as np
from scipy.stats import poisson

def exact_truncation_N(mu: float, epsilon: float, use_half: bool = True, maxN: int = 10_000_000) -> int:
    """
    Return the minimal integer N such that Pr_{X~Poisson(mu)}(X >= N) <= tail_tol,
    where tail_tol = epsilon/2 if use_half is True, otherwise tail_tol = epsilon.
    """
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    tail_tol = epsilon / 2.0 if use_half else epsilon
    if tail_tol >= 1.0:
        return 0
    target = 1.0 - tail_tol
    p = math.exp(-mu)  
    cumulative = p 
    n = 0
    if cumulative >= target:
        return 1  
    while n < maxN:
        n += 1
        p = p * (mu / n)
        cumulative += p
        if cumulative >= target:
            return n + 1
    raise RuntimeError(f"Exceeded maxN={maxN} without meeting tolerance; try increasing maxN")

def find_cutoff_coherent(alpha, epsilon=1e-3):
    """
    Determines truncation N based on Theorem II.1 (Tail-control).
    The tail probability must be <= epsilon / 2 [Source: 47].
    """
    mean_n = np.abs(alpha)**2
    
    threshold = 1 - (epsilon / 2.0)
    
    N_candidate = int(poisson.ppf(threshold, mean_n))
    
    tail_weight = 1.0 - poisson.cdf(N_candidate - 1, mean_n)
    
    while tail_weight > epsilon / 2.0:
        N_candidate += 1
        tail_weight = 1.0 - poisson.cdf(N_candidate - 1, mean_n)
        
    return N_candidate, tail_weight

def exact_truncation_N(mu: float, epsilon: float, use_half: bool = True, maxN: int = 10_000_000) -> int:
    """
    Return the minimal integer N such that Pr_{X~Poisson(mu)}(X >= N) <= tail_tol,
    where tail_tol = epsilon/2 if use_half is True, otherwise tail_tol = epsilon.
    """
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    tail_tol = epsilon / 2.0 if use_half else epsilon
    if tail_tol >= 1.0:
        return 0
    target = 1.0 - tail_tol
    p = math.exp(-mu)  # P(X=0)
    cumulative = p
    n = 0
    if cumulative >= target:
        return 1
    while n < maxN:
        n += 1
        p = p * (mu / n)
        cumulative += p
        if cumulative >= target:
            return n + 1
    raise RuntimeError(f"Exceeded maxN={maxN} without meeting tolerance; try increasing maxN")

def compute_dynamical_cutoff(mu: float,
                             epsilon: float,
                             t: float,
                             g: float,
                             C: float = 1.0,
                             chi: float = None,
                             use_half: bool = True,
                             r_exponent: float = 0.5):
    if r_exponent != 0.5:
        raise ValueError("This helper currently supports r_exponent = 0.5 (JC/Rabi scaling).")

    if chi is None:
        chi = float(g)

    N = exact_truncation_N(mu, epsilon, use_half=use_half)

    p = math.exp(-mu)
    cum = p
    for k in range(1, N):
        p = p * (mu / k)
        cum += p
    tail = 1.0 - cum

    Lambda0 = max(0, N - 1)

    dyn_tol = epsilon / 2.0 if use_half else epsilon

    arg = Lambda0 * chi * t / (dyn_tol if dyn_tol > 0 else 1e-30)

    if arg <= 1.0:
        tilde_cont = float(Lambda0)
    else:

        sqrt_tilde = math.sqrt(float(Lambda0)) + C * float(chi) * float(t) * math.log(arg)

        if sqrt_tilde < math.sqrt(float(Lambda0)):
            sqrt_tilde = math.sqrt(float(Lambda0))
        tilde_cont = sqrt_tilde * sqrt_tilde

    tilde_int = int(math.ceil(tilde_cont))

    if tilde_int < Lambda0:
        tilde_int = Lambda0

    N_final = max(N, tilde_int + 1)  
    return {
        "N": N,
        "tail_prob": tail,
        "Lambda0": Lambda0,
        "tildeLambda_continuous": tilde_cont,
        "tildeLambda_int": tilde_int,
        "N_final": N_final,
        "params": {"mu": mu, "epsilon": epsilon, "t": t, "g": g, "chi": chi, "C": C}
    }