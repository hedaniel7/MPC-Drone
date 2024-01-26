% Lane Keeping Scenario:   CasAdi+MPC
clear all
close all
clc


import casadi.*   %starting CasADi


%%---------------------------------------------------
%                  PROBLEM SET UP
%---------------------------------------------------
%% ---------  parameters ------------
T=0.2; %[s] sampling interval
N=2; % prediction horizon
w_lane=5.25; %the width of the lane
l_vehicle=4.7;  % length of vehicle
w_vehicle=1.83;  % width of vehicle


%% %--------- creating symbols ---------
%% ------------- states ---------------
Vx=SX.sym('Vx');%creating symbols Vx
Vy=SX.sym('Vy');
x=SX.sym('x');
y=SX.sym('y');
states=[x;y;Vx;Vy]; 
n_states=length(states);

%% -------------- controls ---------------
ax=SX.sym('ax');
ay=SX.sym('ay');
controls=[ax;ay];
n_controls=length(controls);

%% -------------- system ----------------
rhs=[Vx;Vy;ax;ay];  % kinematic quantities
f=Function('f',{states,controls},{rhs}); % function f(x,u)
% Note that all function objects in CasADi are multiple matrix-valued input, multiple, matrix-valued output.
U=SX.sym('U',n_controls,N); % decision variables(controls)
P=SX.sym('P',n_states+N*(n_states+n_controls)); % parameters(which include 
%the initial state and the reference along the predicted trajectory
%(reference states and reference controls))
X = SX.sym('X',n_states,(N+1));  % A vector that represents the states over the optimization problem.

%% ---------  cost function---------
%---- weighting matrices for states ----
Q=zeros(4,4);
Q(1,1)=1;  % position for x-direction
Q(2,2)=10;  % position for y-direction
Q(3,3)=100;  % velocity for x-direction
Q(4,4)=1;  % velocity for y-direction


%---- weighting matrices for controls ----
R=zeros(2,2);
R(1,1)=1;  % acceration for x-direction
R(2,2)=0.1;  % acceration for y-direction

%---- initial values for cost function ----
obj=0; % cost function
g=[]; % constraints vector
st=X(:,1);  % initial state
g=[g;st-P(1:4)]; %initial condition constraints


for k=1:N
    st=X(:,k);  
    con=U(:,k);  
    obj=obj+(st-P(6*k-1:6*k+2))'*Q*(st-P(6*k-1:6*k+2))+...
        (con-P(6*k+3:6*k+4))'*R*(con-P(6*k+3:6*k+4)); %caculate cost
    ini_next=X(:,k+1);
    f_value=f(st,con);
    ini_next_euler=st+(T*f_value);
    g=[g;ini_next-ini_next_euler]; %compute constraints
end
%% -----------  Constraints  --------------
% The bounds of position  
x_min=-inf;   x_max=inf;   %m
y_min=w_vehicle/2;  y_max=2*w_lane-w_vehicle/2; %m

% The bounds of velocity 
Vx_min=13;    Vx_max=70;   %m/s
Vy_min=-0.5;  Vy_max=0.5;  %m/s

%The bounds of acceleration
ax_min=-9;  ax_max=6;
ay_min=-0.5;  ay_max=0.5;

%Constraints
args=struct;  
args.lbg(1:4*(N+1))=0;  % -1e-20  % Equality constraints
args.ubg(1:4*(N+1))=0;  % 1e-20  % Equality constraints

args.lbx(1:4:4*(N+1),1)=x_min;  %state x lower bound
args.ubx(1:4:4*(N+1),1)=x_max;  %state x upper bound
args.lbx(2:4:4*(N+1),1)=y_min;  %state y lower bound
args.ubx(2:4:4*(N+1),1)=y_max;  %state y upper bound
args.lbx(3:4:4*(N+1),1)=Vx_min;  %state Vx lower bound
args.ubx(3:4:4*(N+1),1)=Vx_max;  %state Vx upper bound
args.lbx(4:4:4*(N+1),1)=Vy_min;  %state Vy lower bound
args.ubx(4:4:4*(N+1),1)=Vy_max;  %state Vy upper bound

args.lbx(4*(N+1)+1:2:4*(N+1)+2*N,1) = ax_min; %ax lower bound
args.ubx(4*(N+1)+1:2:4*(N+1)+2*N,1) = ax_max; %ax upper bound
args.lbx(4*(N+1)+2:2:4*(N+1)+2*N,1) = ay_min; %ay lower bound
args.ubx(4*(N+1)+2:2:4*(N+1)+2*N,1) = ay_max; %ay upper bound



%% -------------- parameter preparation -------------------
%------  make the decision variables one column vector --------
OPT_variables=[reshape(X,4*(N+1),1);reshape(U,2*N,1)];
nlp_prob=struct('f',obj,'x',OPT_variables,'g',g,'p',P);

opts=struct;
%------   parameter setting for optimal problems
opts.ipopt.max_iter=2000; %ipopt:an open-source primal-dual interior point method which is included in CasADi installations.
opts.ipopt.print_level=0;
opts.print_time=0;
opts.ipopt.acceptable_tol=1e-8;
opts.ipopt.acceptable_obj_change_tol=1e-6;
solver=nlpsol('solver','ipopt',nlp_prob,opts);

%% -------THE SIMULATION LOOP SHOULD START HERE-----------------
%------------ initial values -------------------------
t0=0;
x0=[0;w_lane/2;Vx_min;0];  % initial condition
xs=[inf;0;20;0]; %reference states
xx(:,1)=x0; % xx contains the history of states
t(1)=t0;
u0 = zeros(N,2);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables
sim_tim = 35; % Maximum simulation time

% %% Start MPC
mpciter = 0;
xx1 = [];% solution TRAJECTORY
u_cl=[];
%------------ main simulation loop ------------
loop_start=tic;
while(mpciter < sim_tim / T) % condition for ending the loop
    current_time = mpciter*T;  %get the current time
    args.p(1:4) = x0; % initial condition of the vehicle
    for k=1:N %set the reference to track
        t_predict=current_time+(k-1)*T; % predicted time instant
        x_ref=20*t_predict;
        y_ref=0.5*w_lane;
        Vx_ref=20;
        Vy_ref=0;
        ax_ref=0;
        ay_ref=0;
        args.p(6*k-1:6*k+2)=[x_ref,y_ref,Vx_ref,Vy_ref]; %reference states
        args.p(6*k+3:6*k+4)=[ax_ref,ay_ref]; %desired controls
    end
    %------------intial value of the optimization variables-----------
    args.x0=[reshape(X0',4*(N+1),1);reshape(u0',2*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(4*(N+1)+1:end))',2,N)'; % get controls only from the solution
    xx1(:,1:4,mpciter+1)= reshape(full(sol.x(1:4*(N+1)))',4,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)]; %all controls
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(T, t0, x0, u, f);
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:4*(N+1)))',4,N+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter;
    mpciter = mpciter + 1;
end
%%  calculating time
main_loop_time = toc(loop_start)
average_mpc_time = main_loop_time/(mpciter+1)

%% draw the trajectory and inputs 
Draw_MPC_LaneKeeping(t,xx,xx1,u_cl,xs,N,w_lane,l_vehicle,w_vehicle)


