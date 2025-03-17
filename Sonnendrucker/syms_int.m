syms X;
syms V;
syms k;
syms a;

f = exp(-V^2/2)/sqrt(2*pi)*(1.0 + a*cos(k*X))*V^2;
rho = int(f,V)
qq = 0
