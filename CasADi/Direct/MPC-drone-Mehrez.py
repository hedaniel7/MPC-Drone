{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "initial_id",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:27.048327638Z",
     "start_time": "2024-02-01T14:42:27.045890701Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": "Function(f:(x[12],u[4])->(x_dot[12]) MXFunction)"
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from casadi import *\n",
    "from time import time\n",
    "#from simulation_code import simulate\n",
    "\n",
    "# Paper Cascaded Nonlinear MPC for Realtime Quadrotor Position Tracking. Schlagenhauf\n",
    "\n",
    "# Define system, static/kinematic/dynamic model. Typically Differential (Algebraic) equations\n",
    "\n",
    "# Expression graphs represent a computation in the computer memory. SX, MX\n",
    "\n",
    "\n",
    "Ix = 0.0000166  #Moment of inertia around p_WB_W_x-axis, source: Julian Förster's ETH Bachelor Thesis\n",
    "Iy = 0.0000167  #Moment of inertia around p_WB_W_y-axis, source: Julian Förster's ETH Bachelor Thesis\n",
    "Iz = 0.00000293 #Moment of inertia around p_WB_W_z-axis, source: Julian Förster's ETH Bachelor Thesis\n",
    "m = 0.029 #mass of Crazyflie 2.1\n",
    "g = 9.81\n",
    "\n",
    "Nx = 12\n",
    "Nu = 4\n",
    "Nhoriz = 10\n",
    "T = 5. # time horizon\n",
    "\n",
    "# x(t) = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot]T\n",
    "x = MX.sym(\"x\", Nx)\n",
    "u = MX.sym(\"u\", Nu)\n",
    "\n",
    "\n",
    "x_pos = x[0]  # x-position\n",
    "y = x[1]  # y-position\n",
    "z = x[2]  # z-position\n",
    "phi = x[3]  # phi-angle, Euler angles\n",
    "theta = x[4]  # theta-angle, Euler angles\n",
    "psi = x[5]  # psi-angle, Euler angles\n",
    "x_pos_dot = x[6]  # x velocity\n",
    "y_dot = x[7]  # y velocity\n",
    "z_dot = x[8]  # z velocity\n",
    "phi_dot = x[9]  # phi_dot, angular velocity\n",
    "theta_dot = x[10]  # theta_dot\n",
    "psi_dot = x[11]  # psi-dot\n",
    "\n",
    "thrust = u[0]\n",
    "tau_phi = u[1]\n",
    "tau_theta = u[2]\n",
    "tau_psi = u[3]\n",
    "\n",
    "x_pos_init = 0\n",
    "y_init = 0\n",
    "z_init = 0\n",
    "phi_init = 0\n",
    "theta_init = 0\n",
    "psi_init = 0\n",
    "x_pos_dot_init = 0\n",
    "y_dot_init = 0\n",
    "z_dot_init = 0\n",
    "phi_dot_init = 0\n",
    "theta_dot_init = 0\n",
    "psi_dot_init = 0\n",
    "\n",
    "\n",
    "x_pos_target = 0\n",
    "y_target = 0\n",
    "z_target = 1\n",
    "phi_target = 0\n",
    "theta_target = 0\n",
    "psi_target = 0\n",
    "x_pos_dot_target = 0\n",
    "y_dot_target = 0\n",
    "z_dot_target = 0\n",
    "phi_dot_target = 0\n",
    "theta_dot_target = 0\n",
    "psi_dot_target = 0\n",
    "\n",
    "\n",
    "# x˙(t) = [x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot, x_ddot, y_ddot, z_ddot, phi_ddot, ˙theta_ddot, psi_ddot]T\n",
    "\n",
    "\n",
    "x_pos_ddot = (cos(phi) * sin(phi) * cos(psi) + sin(phi) * sin(phi))  * thrust / m\n",
    "y_ddot = (cos(phi) * sin(phi) * cos(psi) - sin(phi) * sin(phi))  * thrust / m\n",
    "z_ddot = -g + (cos(phi) * cos(theta)) * thrust / m\n",
    "phi_ddot = theta_dot * psi_dot * (Iy - Iz) / (Ix) + tau_phi / Ix\n",
    "theta_ddot = phi_dot * psi_dot * (Iz - Ix) / (Iy) + tau_phi / Iy\n",
    "psi_ddot = theta_dot * phi_dot * (Ix - Iy) / (Iz) + tau_phi / Iz\n",
    "\n",
    "x_dot = vertcat(x_pos_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot)\n",
    "\n",
    "f = Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])\n",
    "\n",
    "f"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "outputs": [],
   "source": [
    "U = MX.sym(\"U\", Nu, Nhoriz) # Decision variables (controls)\n",
    "P = MX.sym('P',Nx + Nx) #parameters (which include the initial state and the reference state)\n",
    "X = MX.sym(\"X\", Nx, Nhoriz + 1) # A vector that represents the states over the optimization problem.\n",
    "\n"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:29.740792821Z",
     "start_time": "2024-02-01T14:42:29.732397666Z"
    }
   },
   "id": "6640b29d3407b8df"
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "outputs": [],
   "source": [
    "J = 0 # Objective function\n",
    "g = []  # constraints vector\n",
    "\n",
    "#Q=diag(MX([100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]))\n",
    "Q = diagcat(100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0)\n",
    "#R=diag(MX([10.0, 10.0, 10.0, 10.0]))\n",
    "R = diagcat(10.0, 10.0, 10.0, 10.0)\n",
    "\n",
    "\n",
    "\n",
    "x_init = P[0:Nx]\n",
    "g = X[:, 0] - P[0:Nx]  # initial condition constraints\n",
    "h = 0.2\n",
    "J = 0\n",
    "\n",
    "for k in range(Nhoriz-1):  \n",
    "    st_ref = P[Nx:2*Nx]\n",
    "    st = X[:, k]\n",
    "    cont = U[:, k]\n",
    "    J += (st - st_ref).T @ Q @ (st - st_ref) + cont.T @ R @ cont\n",
    "    st_next = X[:, k+1]\n",
    "    k1 = f(st, cont)  \n",
    "    k2 = f(st + h / 2 * k1, cont)\n",
    "    k3 = f(st + h / 2 * k2, cont)\n",
    "    k4 = f(st + h * k3, cont)\n",
    "    st_next_RK4 = st + h/6*(k1 + 2*k2 + 2*k3 + k4)  # RK4 integration\n",
    "    g = vertcat(g, st_next - st_next_RK4) # Multiple Shooting\n",
    "\n",
    "\n",
    "\n"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:31.181356954Z",
     "start_time": "2024-02-01T14:42:31.179786171Z"
    }
   },
   "id": "d36e2e151a6df006"
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "outputs": [],
   "source": [
    "OPT_variables = vertcat(\n",
    "    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1\n",
    "    U.reshape((-1, 1))\n",
    ")\n",
    "\n",
    "nlp_prob = {\n",
    "    'f': J,\n",
    "    'x': OPT_variables,\n",
    "    'g': g,\n",
    "    'p': P\n",
    "}\n",
    "\n",
    "opts = {\n",
    "    'ipopt': {\n",
    "        'max_iter': 2000,\n",
    "        'print_level': 0,\n",
    "        'acceptable_tol': 1e-8,\n",
    "        'acceptable_obj_change_tol': 1e-6\n",
    "    },\n",
    "    'print_time': 0\n",
    "}\n",
    "\n",
    "solver = nlpsol('solver', 'ipopt', nlp_prob, opts)\n",
    "\n",
    "lbx = DM.zeros((Nx*(Nhoriz+1) + Nu*Nhoriz, 1))\n",
    "ubx = DM.zeros((Nx*(Nhoriz+1) + Nu*Nhoriz, 1))\n",
    "\n",
    "lbx[0: Nx*(Nhoriz+1): Nx] = -inf     # x lower bound\n",
    "lbx[1: Nx*(Nhoriz+1): Nx] = -inf     # y lower bound\n",
    "lbx[2: Nx*(Nhoriz+1): Nx] = -inf     # z lower bound\n",
    "lbx[3: Nx*(Nhoriz+1): Nx] = -inf     # phi lower bound\n",
    "lbx[4: Nx*(Nhoriz+1): Nx] = -inf     # theta lower bound\n",
    "lbx[5: Nx*(Nhoriz+1): Nx] = -inf     # psi lower bound\n",
    "lbx[6: Nx*(Nhoriz+1): Nx] = -inf     # x_dot lower bound\n",
    "lbx[7: Nx*(Nhoriz+1): Nx] = -inf     # y_dot lower bound\n",
    "lbx[8: Nx*(Nhoriz+1): Nx] = -inf     # z_dot lower bound\n",
    "lbx[9: Nx*(Nhoriz+1): Nx] = -inf     # phi_dot lower bound\n",
    "lbx[10: Nx*(Nhoriz+1): Nx] = -inf     # theta_dot lower bound\n",
    "lbx[11: Nx*(Nhoriz+1): Nx] = -inf     # psi_dot lower bound\n",
    "\n",
    "\n",
    "ubx[0: Nx*(Nhoriz+1): Nx] = inf     # x upper bound\n",
    "ubx[1: Nx*(Nhoriz+1): Nx] = inf     # y upper bound\n",
    "ubx[2: Nx*(Nhoriz+1): Nx] = inf     # z upper bound\n",
    "ubx[3: Nx*(Nhoriz+1): Nx] = inf     # phi upper bound\n",
    "ubx[4: Nx*(Nhoriz+1): Nx] = inf     # theta upper bound\n",
    "ubx[5: Nx*(Nhoriz+1): Nx] = inf     # psi upper bound\n",
    "ubx[6: Nx*(Nhoriz+1): Nx] = inf     # x_dot upper bound\n",
    "ubx[7: Nx*(Nhoriz+1): Nx] = inf     # y_dot upper bound\n",
    "ubx[8: Nx*(Nhoriz+1): Nx] = inf     # z_dot upper bound\n",
    "ubx[9: Nx*(Nhoriz+1): Nx] = inf     # phi_dot upper bound\n",
    "ubx[10: Nx*(Nhoriz+1): Nx] = inf     # theta_dot upper bound\n",
    "ubx[11: Nx*(Nhoriz+1): Nx] = inf     # psi_dot upper bound\n",
    "\n",
    "lbx[Nx*(Nhoriz+1):] = -2                 # v lower bound for all u\n",
    "ubx[Nx*(Nhoriz+1):] = 2                  # v upper bound for all u\n",
    "\n",
    "args = {\n",
    "    'lbg': DM.zeros((Nx*(Nhoriz+1), 1)),  # constraints lower bound\n",
    "    'ubg': DM.zeros((Nx*(Nhoriz+1), 1)),  # constraints upper bound\n",
    "    'lbx': lbx,\n",
    "    'ubx': ubx\n",
    "}"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:33.001373636Z",
     "start_time": "2024-02-01T14:42:32.999940842Z"
    }
   },
   "id": "b7aacdddf1a47610"
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "e2c72a45bc90319a",
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:35.924795643Z",
     "start_time": "2024-02-01T14:42:35.917766265Z"
    }
   },
   "outputs": [],
   "source": [
    "t0 = 0\n",
    "state_init = DM([x_pos_init, y_init, z_init, phi_init, theta_init, psi_init, x_pos_dot_init, y_dot_init, z_dot_init, phi_dot_init, theta_dot_init, psi_dot_init])        # initial state\n",
    "state_target = DM([x_pos_target, y_target, z_target, phi_target, theta_target, psi_target, x_pos_dot_target, y_dot_target, z_dot_target, phi_dot_target, theta_dot_target, psi_dot_target])  # target state\n",
    "\n",
    "# xx = DM(state_init)\n",
    "t = DM(t0)\n",
    "\n",
    "u0 = DM.zeros((Nu, Nhoriz))  # initial control\n",
    "X0 = repmat(state_init, 1, Nhoriz+1)         # initial state full\n",
    "\n",
    "\n",
    "def DM2Arr(dm):\n",
    "    return np.array(dm.full())\n",
    "\n",
    "mpc_iter = 0\n",
    "cat_states = DM2Arr(X0)\n",
    "cat_controls = DM2Arr(u0[:, 0])\n",
    "times = np.array([[0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "outputs": [],
   "source": [
    "def shift_timestep(step_horizon, t0, state_init, u, f):\n",
    "    f_value = f(state_init, u[:, 0])\n",
    "    next_state = DM.full(state_init + (step_horizon * f_value))\n",
    "\n",
    "    t0 = t0 + step_horizon\n",
    "    u0 = horzcat(\n",
    "        u[:, 1:],\n",
    "        reshape(u[:, -1], -1, 1)\n",
    "    )\n",
    "\n",
    "    return t0, next_state, u0"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:39.077276771Z",
     "start_time": "2024-02-01T14:42:39.070135957Z"
    }
   },
   "id": "280fbc0cd575efa8"
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f09dd5144b2a1c01",
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2024-02-01T14:42:45.766521561Z",
     "start_time": "2024-02-01T14:42:45.719009378Z"
    }
   },
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "Error in Function::call for 'solver' [IpoptInterface] at .../casadi/core/function.cpp:1401:\nError in Function::call for 'solver' [IpoptInterface] at .../casadi/core/function.cpp:330:\n.../casadi/core/function_internal.hpp:1644: Input 4 (lbg) has mismatching shape. Got 132-by-1. Allowed dimensions, in general, are:\n - The input dimension N-by-M (here 120-by-1)\n - A scalar, i.e. 1-by-1\n - M-by-N if N=1 or M=1 (i.e. a transposed vector)\n - N-by-M1 if K*M1=M for some K (argument repeated horizontally)\n - N-by-P*M, indicating evaluation with multiple arguments (P must be a multiple of 1 for consistency with previous inputs)",
     "output_type": "error",
     "traceback": [
      "\u001B[0;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[0;31mRuntimeError\u001B[0m                              Traceback (most recent call last)",
      "\u001B[0;32m/tmp/ipykernel_16428/2915659607.py\u001B[0m in \u001B[0;36m?\u001B[0;34m()\u001B[0m\n\u001B[1;32m     10\u001B[0m         \u001B[0mreshape\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mX0\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mNx\u001B[0m\u001B[0;34m*\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mNhoriz\u001B[0m\u001B[0;34m+\u001B[0m\u001B[0;36m1\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0;36m1\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m     11\u001B[0m         \u001B[0mreshape\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mu0\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0mNu\u001B[0m\u001B[0;34m*\u001B[0m\u001B[0mNhoriz\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0;36m1\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m     12\u001B[0m     \u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m     13\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m---> 14\u001B[0;31m     sol = solver(\n\u001B[0m\u001B[1;32m     15\u001B[0m         \u001B[0mx0\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0margs\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0;34m'x0'\u001B[0m\u001B[0;34m]\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m     16\u001B[0m         \u001B[0mlbx\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0margs\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0;34m'lbx'\u001B[0m\u001B[0;34m]\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m     17\u001B[0m         \u001B[0mubx\u001B[0m\u001B[0;34m=\u001B[0m\u001B[0margs\u001B[0m\u001B[0;34m[\u001B[0m\u001B[0;34m'ubx'\u001B[0m\u001B[0;34m]\u001B[0m\u001B[0;34m,\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n",
      "\u001B[0;32m~/anaconda3/envs/CasADi-Jupyter/lib/python3.12/site-packages/casadi/casadi.py\u001B[0m in \u001B[0;36m?\u001B[0;34m(self, *args, **kwargs)\u001B[0m\n\u001B[1;32m  23368\u001B[0m         \u001B[0;32melse\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  23369\u001B[0m           \u001B[0;32mreturn\u001B[0m \u001B[0mtuple\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mret\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  23370\u001B[0m       \u001B[0;32melse\u001B[0m\u001B[0;34m:\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  23371\u001B[0m     \u001B[0;31m# Named inputs -> return dictionary\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0;32m> 23372\u001B[0;31m         \u001B[0;32mreturn\u001B[0m \u001B[0mself\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mcall\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mkwargs\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m",
      "\u001B[0;32m~/anaconda3/envs/CasADi-Jupyter/lib/python3.12/site-packages/casadi/casadi.py\u001B[0m in \u001B[0;36m?\u001B[0;34m(self, *args)\u001B[0m\n\u001B[1;32m  20017\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  20018\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  20019\u001B[0m \u001B[0;34m\u001B[0m\u001B[0m\n\u001B[1;32m  20020\u001B[0m         \"\"\"\n\u001B[0;32m> 20021\u001B[0;31m         \u001B[0;32mreturn\u001B[0m \u001B[0m_casadi\u001B[0m\u001B[0;34m.\u001B[0m\u001B[0mFunction_call\u001B[0m\u001B[0;34m(\u001B[0m\u001B[0mself\u001B[0m\u001B[0;34m,\u001B[0m \u001B[0;34m*\u001B[0m\u001B[0margs\u001B[0m\u001B[0;34m)\u001B[0m\u001B[0;34m\u001B[0m\u001B[0;34m\u001B[0m\u001B[0m\n\u001B[0m",
      "\u001B[0;31mRuntimeError\u001B[0m: Error in Function::call for 'solver' [IpoptInterface] at .../casadi/core/function.cpp:1401:\nError in Function::call for 'solver' [IpoptInterface] at .../casadi/core/function.cpp:330:\n.../casadi/core/function_internal.hpp:1644: Input 4 (lbg) has mismatching shape. Got 132-by-1. Allowed dimensions, in general, are:\n - The input dimension N-by-M (here 120-by-1)\n - A scalar, i.e. 1-by-1\n - M-by-N if N=1 or M=1 (i.e. a transposed vector)\n - N-by-M1 if K*M1=M for some K (argument repeated horizontally)\n - N-by-P*M, indicating evaluation with multiple arguments (P must be a multiple of 1 for consistency with previous inputs)"
     ]
    }
   ],
   "source": [
    "main_loop = time()  # return time in sec\n",
    "while (norm_2(state_init - state_target) > 1e-1) and (mpc_iter * h < 200):\n",
    "    t1 = time()\n",
    "    args['p'] = vertcat(\n",
    "        state_init,    # current state\n",
    "        state_target   # target state\n",
    "    )\n",
    "    # optimization variable current state\n",
    "    args['x0'] = vertcat(\n",
    "        reshape(X0, Nx*(Nhoriz+1), 1),\n",
    "        reshape(u0, Nu*Nhoriz, 1)\n",
    "    )\n",
    "\n",
    "    sol = solver(\n",
    "        x0=args['x0'],\n",
    "        lbx=args['lbx'],\n",
    "        ubx=args['ubx'],\n",
    "        lbg=args['lbg'],\n",
    "        ubg=args['ubg'],\n",
    "        p=args['p']\n",
    "    )\n",
    "\n",
    "    u = reshape(sol['x'][Nx * (Nhoriz + 1):], Nu, Nhoriz)\n",
    "    X0 = reshape(sol['x'][: Nx * (Nhoriz+1)], Nx, Nhoriz+1)\n",
    "\n",
    "    cat_states = np.dstack((\n",
    "        cat_states,\n",
    "        DM2Arr(X0)\n",
    "    ))\n",
    "\n",
    "    cat_controls = np.vstack((\n",
    "        cat_controls,\n",
    "        DM2Arr(u[:, 0])\n",
    "    ))\n",
    "    t = np.vstack((\n",
    "        t,\n",
    "        t0\n",
    "    ))\n",
    "\n",
    "    t0, state_init, u0 = shift_timestep(h, t0, state_init, u, f)\n",
    "\n",
    "    # print(X0)\n",
    "    X0 = horzcat(\n",
    "        X0[:, 1:],\n",
    "        reshape(X0[:, -1], -1, 1)\n",
    "    )\n",
    "\n",
    "    # xx ...\n",
    "    t2 = time()\n",
    "    print(mpc_iter)\n",
    "    print(t2-t1)\n",
    "    times = np.vstack((\n",
    "        times,\n",
    "        t2-t1\n",
    "    ))\n",
    "\n",
    "    mpc_iter = mpc_iter + 1\n",
    "\n",
    "main_loop_time = time()\n",
    "ss_error = norm_2(state_init - state_target)\n",
    "\n",
    "print('\\n\\n')\n",
    "print('Total time: ', main_loop_time - main_loop)\n",
    "print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')\n",
    "print('final error: ', ss_error)\n",
    "\n",
    "# simulate\n",
    "# simulate(cat_states, cat_controls, times, h, Nhoriz,\n",
    "#         np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
