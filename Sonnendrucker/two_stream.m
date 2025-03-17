%% Two stream instability simulation in one-dimensional case.
% Simulation is based on simultaneous numerical solution of Poisson and 
% Vlasov equations. 
%
% Find more information about numerical method and the example below in 
% the original paper - C.Z. Cheng and Georg Knorr "The Integration of the 
% Vlasov Equation in Configuration Space" // J. Comp. Phys. 22 (1976) 
% pp. 330-351 (DOI: 10.1016/0021-9991(76)90053-X)
% and its former improvement givenwrite in E. Sonnendrücker, J. Roche, 
% P. Bertrand, and A. Ghizzo "The Semi-Lagrangian Method for the Numerical 
% Resolution of the Vlasov Equation" // J. Comput. Phys. 149 (1999) 
% pp. 201-220 (DOI: 10.1006/jcph.1998.6148)
function Cheng_Knorr_Sonnerdrucker
clc; close all
%% Problem parameters
% Interpolation type: 'linear' corresponds to first order scheme, 
%                     'spline' Cheng-Knorr & Sonnendrücker cubic spline scheme
inter_type = 'spline';
% Number of cells for uniform phase-space grid
N = 10; M = 10;
% Problem parameters 
k = 0.5; alpha = 0.05; vmax = 2*pi;
% CFL condition
CFL = 3.8;
% End time
t_end = 15.0;
% Periodic structure length in X
L = 2*pi/k;
% Grid definition
x = linspace(0, L, N); v = linspace(-vmax, vmax, M);
dx = x(2)-x(1); dv = v(2)-v(1);
[X, V] = meshgrid(x, v); X = X'; V = V';


%% Initial conditions
f = exp(-V.^2/2)/sqrt(2*pi).*(1.0 + alpha*cos(k*X)).*V.^2;
figure;
surf(X,V,f);
% Apply periodic borders and zero at |v| beyond vmax
f(end, :) = f(2,:); f(1, :) = f(end-1, :);
f(1, :) = 0; f(end, :) = 0;
% Open maximized figure window
figure('units','normalized','outerposition',[0.25 0.25 0.55 0.55]);
set(gcf, 'doublebuffer', 'on');   
% Preallocate memory
E = zeros(size(x));
%% Start main calculation procedure
time = 0;   tic;
while (time<t_end)
    %% Estimate time step using CFL condition
    dt = CFL/(vmax/dx + max(abs(E))/dv);
    if (dt > t_end-time) dt = t_end-time; end
    
    %% Plot EDF f(x,v,t) in phase space
    pcolor(x(2:(end-1)), v(2:(end-1)), f(2:(end-1), 2:(end-1))');
    shading interp;
    axis square;
    drawnow; 
    %% X-coordinate shift at half time step
    x_SHIFT = X - V*0.5*dt;
    x_SHIFT = (x_SHIFT+L).*(x_SHIFT<=0) + (x_SHIFT-L).*(x_SHIFT>=L) + x_SHIFT.*((x_SHIFT>0)&(x_SHIFT<L));  
    f = interp2(X', V', f', x_SHIFT', V', inter_type)'; 
    % Apply periodic boundaries in X-coordinate
    f(N,:) = f(2,:);  f(1,:) = f(N-1,:);
    %% Electrical field strength from exact solution of Poisson's equation 
    %% with periodic border conditions for electric potential
    E = cumtrapz(x, trapz(v, f, 2)) - x';   E = E - mean(mean(E));
    write_poisson_info(x,v,f,E,time); 
    %% V-coordinate shift at full time step
    Vsh = V - repmat(E, 1, M)*dt;
    Vsh = Vsh.*((Vsh<vmax)&(Vsh>=-vmax));
    f = interp2(X', V', f', X', Vsh', inter_type)'; 
    % Boundary conditions
    f(:, 1) = 0; f(:, end) = 0;
    f(end, :) = f(2,:); f(1, :) = f(end-1, :);
  
    %% X-coordinate shift at half time step
    x_SHIFT = X - V*0.5*dt;
    x_SHIFT = (x_SHIFT+L).*(x_SHIFT<=0) + (x_SHIFT-L).*(x_SHIFT>=L) + x_SHIFT.*((x_SHIFT>0)&(x_SHIFT<L));  
    f = interp2(X', V', f', x_SHIFT', V', inter_type)'; 
    % Apply periodic boundaries in X-coordinate
    f(N,:) = f(2,:);  f(1,:) = f(N-1,:);
      
    %% Next time step
    time = time + dt; 
end
toc
% Final EDF plot
pcolor(x(2:(end-1)), v(2:(end-1)), f(2:(end-1), 2:(end-1))');
shading interp;
axis square;
title('Two stream instability', 'FontSize', 18);
colorbar
drawnow;