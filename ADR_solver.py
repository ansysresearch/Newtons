from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from spaces import GRF


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def main():
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    # No reaction
    # g = lambda u: np.zeros_like(u)
    # dg = lambda u: np.zeros_like(u)
    # Reaction
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u


    # Random f
    # rn = 2 * np.random.rand(101, 1) - 1
    # f = lambda x, t: np.tile(rn, (1, len(t)))


    # f0
    f0 = lambda x: 0.9 * np.sin(2 * np.pi * x)
    # f = lambda x, t: f0(x) + 0 * t

    # New f
    # f = lambda x, t: f0(x) + 0.05 + 0 * t
    # New random f
    space = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    features = space.random(1)
    f = lambda x, t: f0(x) + 0.05 * space.eval_u(features, x).T + 0 * t
    u0 = lambda x: np.zeros_like(x)

    Nx, Nt = 101, 101
    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)
    f = f(x[:, None], t)
    np.savetxt("f.dat", f)
    np.savetxt("u.dat", u)

    plt.plot(x, f0(x), label="f0")
    plt.plot(x, f[:, 0], label="f")
    plt.legend()
    plt.show()
    # x = np.repeat(x, Nt)
    # t = np.tile(t, Nx)
    # f = np.repeat(f, Nt)
    # u = np.ravel(u)
    # np.savetxt('data.dat', np.vstack((x, t, f, u)).T)
    # return

    inputs = []
    outputs = []
    for i in range(1, Nx - 1):
        for j in range(1, Nt):
            # inputs.append([f[i, j], u[i, j - 1], u[i - 1, j], u[i + 1, j]])
            inputs.append(
                [
                    f[i, j],
                    u[i - 1, j],
                    u[i - 1, j - 1],
                    u[i, j - 1],
                    u[i + 1, j - 1],
                    u[i + 1, j],
                ]
            )
            outputs.append([u[i, j]])
    # np.savetxt("data_f0.dat", np.hstack((inputs, outputs)))


if __name__ == "__main__":
    main()
