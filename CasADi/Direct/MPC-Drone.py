from casadi import *

# Paper Cascaded Nonlinear MPC for Realtime Quadrotor Position Tracking. Schlagenhauf

# Define system, static/kinematic/dynamic model. Typically Differential (Algebraic) equations

# Expression graphs represent a computation in the computer memory. SX, MX


Ix = 0.0000166  #Moment of inertia around p_WB_W_x-axis, source: Julian Förster's ETH Bachelor Thesis
Iy = 0.0000167  #Moment of inertia around p_WB_W_y-axis, source: Julian Förster's ETH Bachelor Thesis
Iz = 0.00000293 #Moment of inertia around p_WB_W_z-axis, source: Julian Förster's ETH Bachelor Thesis
m = 0.029 #mass of Crazyflie 2.1
g = 9.81

Nx = 12
Nu = 4
Nhoriz = 10

T = 5. # Time horizon

# x(t) = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot]T
X = MX.sym("x",Nx)
U = MX.sym("u",Nu)


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


# x˙(t) = [x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot, x_ddot, y_ddot, z_ddot, phi_ddot, ˙theta_ddot, psi_ddot]T
x_dot = vertcat(x_pos_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot)

x_pos_ddot = (cos(phi) * sin(phi) * cos(psi) + sin(phi) * sin(phi))  * thrust / m
y_ddot = (cos(phi) * sin(phi) * cos(psi) - sin(phi) * sin(phi))  * thrust / m
z_ddot = -g + (cos(phi) * cos(theta)) * thrust / m
phi_ddot = theta_dot * psi_dot * (Iy - Iz) / (Ix) + tau_phi / Ix
theta_ddot = phi_dot * psi_dot * (Iz - Ix) / (Iy) + tau_phi / Iy
psi_ddot = theta_dot * phi_dot * (Ix - Iy) / (Iz) + tau_phi / Iz


f = Function('f', [X, U], [x_dot], ['x','u'],['x_dot'])


## Another ingredient we can use towards MPC  is a time-integration method

T = 10 # time horizon
N = 20 # Number of control intervals

## Create an integrator


#dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
#F = integrator('F', 'cvodes', dae, 0, T/N)












# Define problem based on system. Initial value/Integration/Rootfinding Problems/
# Nonlinear constrained optimization

opti = Opti()
x = opti.variable( Nx, Nhoriz + 1)
u = opti.variable(Nu, Nhoriz)


# Expression graphs -create-> create functions

# Solve/deploy problem
# Numerical backends, 3rd-party solvers
# Ipopt

#functions -evaluate->







# MISC
