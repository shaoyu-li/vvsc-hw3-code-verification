import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Exact solution
# ============================================================
def exact_solution(x, t, alpha):
    """
    Exact analytical solution:
        T(x,t) = sin(pi*x) * exp(-alpha*pi^2*t)
    """
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


# ============================================================
# 2. Explicit finite-difference solver (FTCS)
# ============================================================
def solve_heat_equation_ftcs(N, alpha, t_final, cfl=0.4):
    """
    Solve:
        T_t = alpha * T_xx
    on x in [0,1], t > 0
    with:
        T(0,t)=0, T(1,t)=0
        T(x,0)=sin(pi*x)

    Parameters
    ----------
    N : int
        Number of spatial intervals
    alpha : float
        Thermal diffusivity
    t_final : float
        Final simulation time
    cfl : float
        Safety factor for explicit stability:
            dt = cfl * dx^2 / alpha
        Must satisfy cfl <= 0.5 for FTCS stability

    Returns
    -------
    x : ndarray
        Grid points
    T : ndarray
        Numerical solution at final time
    dt : float
        Time step used
    n_steps : int
        Number of time steps
    """

    # Spatial grid
    L = 1.0
    dx = L / N
    x = np.linspace(0.0, L, N + 1)

    # Stable explicit time step: dt = O(dx^2)
    dt_est = cfl * dx**2 / alpha

    # Adjust dt so final time is reached exactly
    n_steps = int(np.ceil(t_final / dt_est))
    dt = t_final / n_steps

    # Recompute effective lambda
    lam = alpha * dt / dx**2
    if lam > 0.5:
        raise ValueError(
            f"FTCS unstable: alpha*dt/dx^2 = {lam:.6f} > 0.5. "
            f"Choose a smaller CFL."
        )

    # Initial condition
    T = exact_solution(x, 0.0, alpha)

    # Time marching
    for _ in range(n_steps):
        T_new = T.copy()

        # Interior update
        T_new[1:N] = T[1:N] + lam * (T[2:N+1] - 2.0 * T[1:N] + T[0:N-1])

        # Dirichlet BCs from exact solution
        T_new[0] = 0.0
        T_new[N] = 0.0

        T = T_new

    return x, T, dt, n_steps


# ============================================================
# 3. Error computation
# ============================================================
def compute_error_metrics(x, T_num, t_final, alpha):
    """
    Compute local error and global error norms.
    """
    T_ex = exact_solution(x, t_final, alpha)
    e = T_num - T_ex

    # Number of nodal points
    npts = len(x)

    L1 = np.sum(np.abs(e)) / npts
    L2 = np.sqrt(np.sum(e**2) / npts)
    Linf = np.max(np.abs(e))

    # System Response Quantity (SRQ): max temperature at final time
    srq_num = np.max(T_num)
    srq_ex = np.max(T_ex)
    srq_err = abs(srq_num - srq_ex)

    return {
        "error_local": e,
        "T_exact": T_ex,
        "L1": L1,
        "L2": L2,
        "Linf": Linf,
        "SRQ_num": srq_num,
        "SRQ_exact": srq_ex,
        "SRQ_error": srq_err,
    }


# ============================================================
# 4. Observed order of accuracy
# ============================================================
def observed_orders(errors, refinement_ratio):
    """
    Compute observed order using
        p = ln(E_coarse / E_fine) / ln(r)

    Returns a NumPy array of length len(errors)-1.
    """
    p_list = []
    for i in range(len(errors) - 1):
        e_coarse = errors[i]
        e_fine = errors[i + 1]

        if e_coarse <= 0.0 or e_fine <= 0.0:
            p_list.append(np.nan)
        else:
            p = np.log(e_coarse / e_fine) / np.log(refinement_ratio)
            p_list.append(p)

    return np.array(p_list)


# ============================================================
# 5. Main verification study
# ============================================================
def run_verification():
    # Problem parameters
    alpha = 1.0
    t_final = 0.05

    # Systematically refined meshes
    N_values = [10, 20, 40, 80]
    r = 2.0

    # Storage
    h_values = []
    dt_values = []
    nstep_values = []

    L1_errors = []
    L2_errors = []
    Linf_errors = []
    SRQ_errors = []

    solutions = {}

    # Loop over meshes
    for N in N_values:
        x, T_num, dt, n_steps = solve_heat_equation_ftcs(
            N=N,
            alpha=alpha,
            t_final=t_final,
            cfl=0.4
        )

        metrics = compute_error_metrics(x, T_num, t_final, alpha)

        h = 1.0 / N

        h_values.append(h)
        dt_values.append(dt)
        nstep_values.append(n_steps)

        L1_errors.append(metrics["L1"])
        L2_errors.append(metrics["L2"])
        Linf_errors.append(metrics["Linf"])
        SRQ_errors.append(metrics["SRQ_error"])

        solutions[N] = {
            "x": x,
            "T_num": T_num,
            "T_exact": metrics["T_exact"],
            "error_local": metrics["error_local"],
        }

    # Convert to arrays
    h_values = np.array(h_values)
    dt_values = np.array(dt_values)
    nstep_values = np.array(nstep_values)

    L1_errors = np.array(L1_errors)
    L2_errors = np.array(L2_errors)
    Linf_errors = np.array(Linf_errors)
    SRQ_errors = np.array(SRQ_errors)

    # Observed orders
    p_L1 = observed_orders(L1_errors, r)
    p_L2 = observed_orders(L2_errors, r)
    p_Linf = observed_orders(Linf_errors, r)
    p_SRQ = observed_orders(SRQ_errors, r)

    # ========================================================
    # Print tables
    # ========================================================
    print("=" * 84)
    print("Grid Refinement Study")
    print("=" * 84)
    print(
        f"{'N':>8} {'h':>14} {'dt':>14} {'n_steps':>10} "
        f"{'L1':>14} {'L2':>14} {'Linf':>14} {'SRQ err':>14}"
    )
    for i, N in enumerate(N_values):
        print(
            f"{N:8d} {h_values[i]:14.6e} {dt_values[i]:14.6e} {nstep_values[i]:10d} "
            f"{L1_errors[i]:14.6e} {L2_errors[i]:14.6e} {Linf_errors[i]:14.6e} {SRQ_errors[i]:14.6e}"
        )

    print("\n" + "=" * 84)
    print("Observed Orders of Accuracy")
    print("=" * 84)
    print(f"{'Pair':>12} {'p(L1)':>12} {'p(L2)':>12} {'p(Linf)':>12} {'p(SRQ)':>12}")
    for i in range(len(N_values) - 1):
        pair = f"{N_values[i]}->{N_values[i+1]}"
        print(
            f"{pair:>12} {p_L1[i]:12.6f} {p_L2[i]:12.6f} "
            f"{p_Linf[i]:12.6f} {p_SRQ[i]:12.6f}"
        )

    # ========================================================
    # Plot 1: numerical vs exact solution on finest mesh
    # ========================================================
    N_fine = N_values[-1]
    x_fine = solutions[N_fine]["x"]
    T_num_fine = solutions[N_fine]["T_num"]
    T_ex_fine = solutions[N_fine]["T_exact"]

    plt.figure(figsize=(7, 5))
    plt.plot(x_fine, T_ex_fine, label="Exact solution")
    plt.plot(x_fine, T_num_fine, "o", markersize=4, label="Numerical solution")
    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.title(f"Numerical vs Exact Solution at t = {t_final}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("solution_comparison.png", dpi=300)

    # ========================================================
    # Plot 2: error norms and SRQ error vs h
    # ========================================================
    plt.figure(figsize=(7, 5))
    plt.loglog(h_values, L1_errors, "o-", label=r"$L_1$")
    plt.loglog(h_values, L2_errors, "s-", label=r"$L_2$")
    plt.loglog(h_values, Linf_errors, "^-", label=r"$L_\infty$")
    plt.loglog(h_values, SRQ_errors, "d-", label="SRQ error")
    plt.xlabel("h")
    plt.ylabel("Error")
    plt.title("Discretization Error Norms and SRQ Error vs Grid Spacing")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("error_norms_vs_h.png", dpi=300)

    # ========================================================
    # Plot 3: observed order vs mesh pair
    # ========================================================
    x_order = np.arange(1, len(N_values))
    labels = [f"{N_values[i]}→{N_values[i+1]}" for i in range(len(N_values) - 1)]

    plt.figure(figsize=(7, 5))
    plt.plot(x_order, p_L1, "o-", label=r"$p(L_1)$")
    plt.plot(x_order, p_L2, "s-", label=r"$p(L_2)$")
    plt.plot(x_order, p_Linf, "^-", label=r"$p(L_\infty)$")
    plt.plot(x_order, p_SRQ, "d-", label=r"$p(SRQ)$")
    plt.axhline(y=2.0, linestyle="--", label="Formal order = 2")
    plt.xticks(x_order, labels)
    plt.xlabel("Mesh pair")
    plt.ylabel("Observed order")
    plt.title("Observed Order of Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("observed_order.png", dpi=300)

    # ========================================================
    # Plot 4: local error on finest mesh
    # ========================================================
    e_fine = solutions[N_fine]["error_local"]

    plt.figure(figsize=(7, 5))
    plt.plot(x_fine, e_fine, "o-")
    plt.xlabel("x")
    plt.ylabel("Local error")
    plt.title(f"Local Discretization Error on Finest Mesh (N={N_fine})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("local_error_finest_mesh.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    run_verification()