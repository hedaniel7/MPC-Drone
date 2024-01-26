# Car race along a track
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.
#
# For more information see: http://labs.casadi.org/OCP
from casadi import *
from pylab import plot, step, figure, legend, show, spy

# setting matrix_weights' variables
Q_x = 100
Q_y = 100
Q_theta = 2000
R1 = 1
R2 = 1
R3 = 1
R4 = 1

# control symbolic variables
f_t = ca.SX.sym('f_t')
tau_x = ca.SX.sym('V_b')
tau_y = ca.SX.sym('V_c')
tau_z = ca.SX.sym('V_d')
controls = ca.vertcat(
    f_t,
    tau_x,
    tau_y,
    tau_z
)

step_horizon = 0.1  # time between steps in seconds
N = 10              # number of look ahead steps
N = 100 # number of control intervals
sim_time = 200      # simulation time

Ix = 0.0000166  # Moment of inertia around p_WB_W_x-axis, source: Julian Förster's ETH Bachelor Thesis
Iy = 0.0000167  # Moment of inertia around p_WB_W_y-axis, source: Julian Förster's ETH Bachelor Thesis
Iz = 0.00000293  # Moment of inertia around p_WB_W_z-axis, source: Julian Förster's ETH Bachelor Thesis
mass = 0.029  # Mass of the quadrotor, source: Julian Förster's ETH Bachelor Thesis
g = 9.81     # Acceleration due to gravity


#      Rotation   angular_vel  vel   position
# roll, pitch, yaw, p, q, r, u, v, w, x, y, z

###################################

opti = Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(2,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]
U = opti.variable(1,N)   # control trajectory (throttle)
T = opti.variable()      # final time

# ---- objective          ---------
opti.minimize(T) # race in minimal time

# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[1],u-x[1]) # dx/dt = f(x,u)

dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
limit = lambda pos: 1-sin(2*pi*pos)/2
opti.subject_to(speed<=limit(pos))   # track speed limit
opti.subject_to(opti.bounded(0,U,1)) # control is limited

# ---- boundary conditions --------
opti.subject_to(pos[0]==0)   # start at position 0 ...
opti.subject_to(speed[0]==0) # ... from stand-still 
opti.subject_to(pos[-1]==1)  # finish line at position 1

# ---- misc. constraints  ----------
opti.subject_to(T>=0) # Time must be positive

# ---- initial values for solver ---
opti.set_initial(speed, 1)
opti.set_initial(T, 1)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

# ---- post-processing        ------


plot(sol.value(speed),label="speed")
plot(sol.value(pos),label="pos")
plot(limit(sol.value(pos)),'r--',label="speed limit")
step(range(N),sol.value(U),'k',label="throttle")
legend(loc="upper left")

figure()
spy(sol.value(jacobian(opti.g,opti.x)))
figure()
spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

show()