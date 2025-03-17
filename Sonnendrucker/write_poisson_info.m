function [] = write_poisson_info(x,v,f,E,time)
rho = trapz(v, f, 2);
fn= sprintf("x_%10.3e.txt",time);

dlmwrite(fn,x,'delimiter','\n','precision',15);
fn= sprintf("rho_%10.3e.txt",time);
dlmwrite(fn,rho,'delimiter','\n','precision',15);
fn= sprintf("E_%10.3e.txt",time);
dlmwrite(fn,E,'delimiter','\n','precision',15);

fn= sprintf("f_%10.3e.txt",time);
dlmwrite(fn,f,'delimiter','\n','precision',15);


end