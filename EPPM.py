import numpy as np
from scipy.linalg import expm               # matrix exponential
from scipy.optimize import minimize         # Nelder-Mead / BFGS
import pandas as pd

# Solve for c by Newton–Raphson method.
def solve_c(m, v, b):
    L = np.log(m/b + 1)
    target = v / (m + b)
    if target >= m / b:               # c >= 1  -> inadmissible
        return None
    t = target / L                    # solve g(x) = (e^x - 1)/x = t
    x = L                             # start at c = 1
    for _ in range(100):
        ex = np.exp(x)
        g  = (ex - 1) / x
        gp = (x*ex - ex + 1) / x**2
        step = (g - t) / gp
        x -= step
        if abs(step) < 1e-10:
            break
    return (x / L + 1) / 2

# Solve for a
def solve_a(m, b, c):                 # eq. (5)
    if c < 1 - 1e-8:
        return ((m + b)**(1 - c) - b**(1 - c)) / (1 - c)
    else:
        return np.log(1 + m / b)

# Optimization for the log likelihood. Definition of the negative log likelihood.
# Data is organized as (y_i, Xm_i, Xv_i)
def negative_loglik(theta, Xm, Xv, y): # theta will be optimized later.
    pm, pv = Xm.shape[1], Xv.shape[1]
    beta_m = theta[:pm]
    beta_v = theta[pm:pm + pv]
    eta = theta[pm + pv]
    b = np.exp(eta)                      # eta = log b
    ll = 0.0
    n = max(y) # truncation at maximal y element
    e_0 = np.zeros(n+1)
    e_0[0] = 1
    for i in range(len(y)):
        m = np.exp(np.dot(beta_m, Xm[i]))   # log link for mean
        v = np.exp(np.dot(beta_v, Xv[i]))
        c = solve_c(m, v, b)
        if c is None or c > 1:
            return 1e10               # penalty: inadmissible region
        a = solve_a(m, b, c)
        lam = a * (b + np.arange(n+1))**c
        Q = -np.diag(lam) + np.diag(lam[:-1], 1)
        p = (e_0 @ expm(Q))
        ll += np.log(p[y[i]])
    return -ll


if __name__ == '__main__':
    # Data preparation
    raw = {
        0.0: [27, 30, 29, 31, 16, 15, 18, 17, 14, 27],
        1.5625: [32, 35, 32, 26, 18, 29, 27, 16, 35, 13],
        3.125: [39, 30, 33, 33, 36, 33, 33, 27, 38, 44],
        6.25: [27, 34, 36, 34, 31, 27, 33, 31, 33, 31],
        12.5: [10, 13, 7, 7, 7, 10, 10, 16, 12, 2],
    }
    power = 2
    rows = []
    for conc, counts in raw.items():
        for c in counts:  # one row per single count
            rows.append({
                "y": c,  # <-- single count  y_i
                "intercept": 1.0,
                "concentration": conc,
                "power_concentration": conc ** power,
            })
    df = pd.DataFrame(rows)
    cols = ["intercept", "concentration", "power_concentration"]
    Xm = df[cols].to_numpy(dtype=float)  # 50 x 3
    Xv = Xm.copy()  # Xv == Xm
    y = df["y"].to_numpy(dtype=int)

    opts = dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-8)

    # Several initial guess is used below.
    # One: initial guess is from log linear Poisson model whose fitting was made in R
    # However, log linear Poisson model makes b NOT to be identifiable.
    b = [0, 0.5, -0.5]
    theta = []
    for i in b:
        theta0 = np.array([3.087, 0.193, -0.021, 3.087, 0.193, -0.021, i])
        res = minimize(negative_loglik, theta0, args=(Xm, Xv, y), method="Nelder-Mead", options=opts)
        theta_hat = minimize(negative_loglik, res.x, args=(Xm, Xv, y), method="Nelder-Mead", options=opts)
        print(f"success={theta_hat.success}")
        theta.append(theta_hat.x)
    print(theta)
    # b = 0, 0.5, -0.5: theta_hat = [ 3.1406405   0.17338906 -0.01961149  4.43847418 -0.49242957  0.02772806 -0.14272941]
    # similar to the results in that paper.
    # b = 1 or -1 -> failed to converge.

    # Two: initial guess is from regression respectively on sample mean and variance (log transformed)
    theta01 = np.array([3.09, 0.19, -0.02, 4.25, -0.45, 0.03, 0])
    res2 = minimize(negative_loglik, theta01, args=(Xm, Xv, y), method="Nelder-Mead", options=opts)
    theta_hat_01 = minimize(negative_loglik, res2.x, args=(Xm, Xv, y), method="Nelder-Mead", options=opts)
    print(f"success={theta_hat_01.success}")
    print(theta_hat_01.x)
    # theta_hat_01 = [ 3.14064049  0.17338906 -0.01961149  4.43847411 -0.49242954  0.02772805 -0.14272877]





