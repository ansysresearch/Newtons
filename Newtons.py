from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import linalg as LA
from spaces import GRF
import matplotlib
import time

iteration_num = 10

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

def construct_data(f, u):
    Nx, Nt = f.shape
    inputs = []
    outputs = []
    for i in range(1, Nx - 1):
        for j in range(1, Nt):
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
    return np.array(inputs), np.array(outputs)
    
def train_step(x_train, u_p):
    with tf.GradientTape() as tape:
        tape.watch(x_train)
        predictions = model_iteration(x_train)
    
    J = tf.squeeze(tape.jacobian(predictions, x_train[1]))
    J_dia = tf.linalg.diag_part(J)
    J_ = tf.linalg.diag(J_dia)
    J_inv = tf.linalg.inv(J_)
    
    u_n = tf.cast(u_p, 'float32') - tf.linalg.matmul(tf.cast(J_inv, 'float32'), predictions)
    return u_n, J, J_inv

activation = tf.keras.activations.relu
inputA = tf.keras.layers.Input(shape=(6,))
inputB = tf.keras.layers.Input(shape=(1,))

inputs = tf.keras.layers.Concatenate(axis=1)([inputA, inputB])
x = tf.keras.layers.Dense(8, activation=activation)(inputs)
x = tf.keras.layers.Dense(1)(x)
model_iteration = tf.keras.Model(inputs=[inputA, inputB], outputs=x)

loss_object = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')

checkpoint_filepath = './checkpoint/cp.ckpt'
model_iteration.load_weights(checkpoint_filepath)

num_study = 1
for case_num in range(num_study):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1

    ## Coefficient definition
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)

    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u

    f0 = lambda x: 0.9 * np.sin(2 * np.pi * x)
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

    print('Finished generating testing GRF f')


    Nx, Nt = 101, 101
    f = np.loadtxt("f.dat")
    f0 = np.loadtxt("f_initial.dat")
    f -= f0
    u_true = np.loadtxt("u.dat")
    ts = time.time()
    errs = []
    max_errs = []
    u0 = np.loadtxt("u_initial.dat")
    u = np.zeros_like(u0)
    sln_cache = np.zeros((9900, 2))
    
    for iter_num in range(iteration_num):
        inputs_f = []
        for i in range(1, Nx - 1):
            for j in range(1, Nt):
                inputs_f.append(
                    [
                        f[i, j],
                        u[i - 1, j],
                        u[i - 1, j - 1],
                        u[i, j - 1],
                        u[i + 1, j - 1],
                        u[i + 1, j],
                    ]
                )
        inputs_f = np.array(inputs_f) * np.array([0.01, 1, 1, 1, 1, 1])
        prev_sln = sln_cache[:, iter_num%2-1][:,None]
        next_sln, J, J_inv = train_step([tf.convert_to_tensor(inputs_f), tf.convert_to_tensor(prev_sln)], tf.convert_to_tensor(prev_sln))
        print('J is: ', J)
        print('J_inv is: ', J_inv)
        k = 0
        for i in range(1, Nx - 1):
            for j in range(1, Nt - 1):
                u[j, i] = next_sln[k, 0].numpy()
                k += 1
                
        use_delta = True       
        if not use_delta:
            err = np.linalg.norm((u_true - u).flatten()) / np.linalg.norm(u_true.flatten())
        else:
            err = np.linalg.norm((u_true - u - u0).flatten()) / np.linalg.norm(u_true.flatten())
        errs.append(err)     
        
        mse = tf.math.reduce_mean(tf.square(tf.subtract(next_sln, prev_sln)))
        print(f'iter:{iter_num}, sln_err: {err}, cur_next_mse: {mse}')    
        sln_cache[:,iter_num%2:iter_num%2+1] = next_sln.numpy()
        
    print("One-shot took %f s\n" % (time.time() - ts))