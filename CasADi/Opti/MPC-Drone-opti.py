# Drone Position control with MPC
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.

from casadi import *
import numpy as np

# Paper Cascaded Nonlinear MPC for Realtime Quadrotor Position Tracking. Schlagenhauf
# Paper Non-Linear Model Predictive Control Using CasADi Package for Trajectory Tracking of Quadrotor. Elhesasy

# 1) Define system, static/kinematic/dynamic model. Typically Differential (Algebraic) equations
# Expression graphs represent a computation in the computer memory. SX, MX

opti = Opti()  # Optimization problem

Ix = 0.0000166  # Moment of inertia around p_WB_W_x-axis, source: Julian Förster's ETH Bachelor Thesis
Iy = 0.0000167  # Moment of inertia around p_WB_W_y-axis, source: Julian Förster's ETH Bachelor Thesis
Iz = 0.00000293  # Moment of inertia around p_WB_W_z-axis, source: Julian Förster's ETH Bachelor Thesis
m = 0.029  # mass of Crazyflie 2.1
g = 9.81

Nx = 12
Nu = 4
Nhoriz = 10

# ---- decision variables ---------


# x(t) = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot]T
X = opti.variable(Nx, Nhoriz + 1)
U = opti.variable(Nu, Nhoriz)

xref = MX.zeros(Nx,1)
xref[2] = 1

Xref = repmat(xref, 1, Nhoriz + 1)

print(Xref[0])

# repmat(v, n, m): Repeat expression v
# n times vertically and m times horizontally.
# repmat(SX(3), 2, 1) will create a 2 - by - 1 matrix with all elements 3.

x_pos = X[0]  # x-position
y = X[1]  # y-position
z = X[2]  # z-position
phi = X[3]  # phi-angle, Euler angles
theta = X[4]  # theta-angle, Euler angles
psi = X[5]  # psi-angle, Euler angles
x_pos_dot = X[6]  # x velocity
y_dot = X[7]  # y velocity
z_dot = X[8]  # z velocity
phi_dot = X[9]  # phi_dot, angular velocity
theta_dot = X[10]  # theta_dot
psi_dot = X[11]  # psi-dot

thrust = U[0]
tau_phi = U[1]
tau_theta = U[2]
tau_psi = U[3]

x_pos_ddot = (cos(phi) * sin(phi) * cos(psi) + sin(phi) * sin(phi)) * thrust / m
y_ddot = (cos(phi) * sin(phi) * cos(psi) - sin(phi) * sin(phi)) * thrust / m
z_ddot = -g + (cos(phi) * cos(theta)) * thrust / m
phi_ddot = theta_dot * psi_dot * (Iy - Iz) / (Ix) + tau_phi / Ix
theta_ddot = phi_dot * psi_dot * (Iz - Ix) / (Iy) + tau_phi / Iy
psi_ddot = theta_dot * phi_dot * (Ix - Iy) / (Iz) + tau_phi / Iz

# x˙(t) = [x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot, x_ddot, y_ddot, z_ddot, phi_ddot, ˙theta_ddot, psi_ddot]T
x_dot = vertcat(x_pos_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot)



# 2) Define problem based on system. Initial value/Integration/Rootfinding Problems/
# Nonlinear constrained optimization

# Expression graphs -create-> create functions

f = Function('f', [X, U], [x_dot], ['x', 'u'], ['x_dot'])

# ---- objective          ---------


Q = diag(MX([100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
R = diag(MX([10.0, 10.0, 10.0, 10.0]))

# Add the objective function to the optimization problem
objective = 0  # Initialize the objective function

# Loop over the control horizon
for k in range(Nhoriz):
    # State tracking cost
    state_error = X[:, k] - Xref[:, k]  # State deviation
    objective += state_error.T @ Q @ state_error  # Quadratic cost for state deviation

    # Control effort cost
    if k < Nhoriz - 1:  # No control input for the last stage
        control_error = U[:, k]  # Assuming zero as the reference for control inputs
        objective += control_error.T @ R @ control_error  # Quadratic cost for control effort

# Terminal cost
terminal_error = X[:, Nhoriz] - Xref[:, Nhoriz]  # Terminal state deviation
objective += terminal_error.T @ Q @ terminal_error  # Quadratic cost for terminal state deviation

# Solve/deploy problem
# Numerical backends, 3rd-party solvers
# Ipopt

opti.minimize(objective)  # Set the objective in the optimization problem

# ---- dynamic constraints --------

# Dynamics constraints and other constraints will be added here



T = 1

dt = T / Nhoriz  # length of a control interval
for k in range(Nhoriz):  # loop over control intervals
    # Runge-Kutta 4 integration
    k1 = f(X[:, k], U[:, k])
    k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = f(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

# ---- path constraints -----------
#opti.subject_to(speed <= limit(pos))  # track speed limit
#opti.subject_to(opti.bounded(0, U, 1))  # control is limited

# ---- boundary conditions --------
#opti.subject_to(pos[0] == 0)  # start at position 0 ...
#opti.subject_to(speed[0] == 0)  # ... from stand-still
#opti.subject_to(pos[-1] == 1)  # finish line at position 1

# ---- initial values for solver ---
opti.set_initial(x_pos, 0)
opti.set_initial(y, 0)
opti.set_initial(z, 0)
opti.set_initial(phi, 0)
opti.set_initial(theta, 0)
opti.set_initial(psi, 0)
opti.set_initial(x_pos_dot, 0)
opti.set_initial(y_dot, 0)
opti.set_initial(z_dot, 0)
opti.set_initial(phi_dot, 0)
opti.set_initial(theta_dot, 0)
opti.set_initial(psi_dot, 0)

# ---- solve NLP              ------

# ---- Setup solver  -----
opts = {'ipopt': {'print_level': 1}, 'print_time': 0}
opti.solver('ipopt', opts)

# ---- Solve the optimization problem -----
sol = opti.solve()

# Now you would extract the optimal trajectory and control inputs
optimal_x = sol.value(X)
optimal_u = sol.value(U)





# ---- post-processing        ------
from pylab import plot, step, figure, legend, show, spy



# MISC
