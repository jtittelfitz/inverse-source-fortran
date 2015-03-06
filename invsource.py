#f2py -c --fcompiler=gnu95 --f90flags=-ffree-form -m PDE PDE.f90

import PDE
import numpy as np
import scipy as sp
import pylab as pl
from matplotlib import animation
import Image


scale = 2
movie = True; write = False; show = False

h = 0.0125*scale; k = 0.0025*scale
r_omega = 2.; r_Rn = 5.
N_grid_omega = 2*int(r_omega/h) + 1; N_grid_Rn = 2*int(r_Rn/h) + 1
r_Rn = h*(N_grid_Rn - 1)/2
T = 4; T_grid = int(T/k) + 1
L = (N_grid_Rn - N_grid_omega)/2
R = N_grid_Rn - (N_grid_Rn - N_grid_omega)/2
zero_tol = 1.0e-4 # criteria for determining support of a source
nonzero_tol = 0.5;

# data simulation parameters
num_sources =2
t_0 = int(T_grid / 4)

# number of iterations for TAT
num_iter = 1

# 
zeros_Rn = np.zeros((N_grid_Rn,N_grid_Rn)); zeros_omega = np.zeros((N_grid_omega,N_grid_omega))

# initialize grid
c = np.zeros((N_grid_Rn,N_grid_Rn)); boundary_mask = np.zeros((N_grid_omega,N_grid_omega))
(c,boundary_mask) = PDE.initialize(c,boundary_mask,h,r_Rn,r_omega)

##
## Data simulation: initial data or source problem -> boundary data
##
print "Beginning data simulation"
Lambda = np.zeros((N_grid_omega,N_grid_omega,T_grid))
force = np.zeros((N_grid_Rn,N_grid_Rn,T_grid))
temp = np.zeros((N_grid_omega,N_grid_omega,T_grid))
g = np.zeros((N_grid_Rn,N_grid_Rn))

t = [10, 50] #np.random.randint(t_0,size=num_sources) + 1

for t_j in t:
     x_0 = -1 + (t_j - 10)*0.05 #np.random.uniform(-r_omega/4,r_omega/4)
     y_0 = -1 + (t_j - 10)*0.05 #np.random.uniform(-r_omega/4,r_omega/4)
     r = 0.25 #np.random.uniform(r_omega/8,r_omega/4)
     print "Source:",x_0,y_0,r,t_j
     g = PDE.circle(g,x_0,y_0,r,h,r_Rn)
     if show:
          print "Showing true source"
          pl.imshow(g)
          pl.show()
     temp = PDE.forward(temp,zeros_Rn,g,c,h,k,boundary_mask)
#     temp = PDE.forwardmovie(temp,zeros_Rn,g,c,h,k)
          
     Lambda[:,:,t_j:T_grid] = Lambda[:,:,t_j:T_grid] + temp[:,:,0:T_grid-t_j]
print "Done with data simulation"

##
## "TAT" reversal: solve elliptic, reverse problem; iterate (forward, elliptic, reverse)
##

c = PDE.newspeed(c,h,r_Rn,r_omega)

phi = np.zeros((N_grid_omega,N_grid_omega))
phi = PDE.elliptic(phi,Lambda,c,h,boundary_mask)

if show:
     print "Solution of elliptic equation"
     pl.imshow(phi)
     pl.show()

if num_iter == 1:
     v = np.zeros((N_grid_omega,N_grid_omega,T_grid))
     v = PDE.reversemovie(v,Lambda,phi,zeros_omega,c,h,k,boundary_mask)
else:     
     f = np.zeros((N_grid_Rn,N_grid_Rn))
     solution = np.zeros((N_grid_omega,N_grid_omega))
     v = np.zeros((N_grid_omega,N_grid_omega))
     v = PDE.reverse(v,Lambda,phi,zeros_omega,c,h,k,boundary_mask)
     solution = v     
     f[L:R,L:R] = v
     if show: 
          pl.imshow(solution); pl.show()
          
     for n in range(1,num_iter):
          Lambda = PDE.forward(Lambda,f,zeros_Rn,c,h,k)
          phi = PDE.elliptic(phi,Lambda,c,h,boundary_mask)
          if n == num_iter-1:
               v = np.zeros((N_grid_omega,N_grid_omega,T_grid))
               v = PDE.reversemovie(v,Lambda,phi,zeros_omega,c,h,k,boundary_mask)
          else:
               v = PDE.reverse(v,Lambda,phi,zeros_omega,c,h,k,boundary_mask)
               # f - v(0) is next term of Neumann series and initial data for next iteration     
               solution = solution + f[L:R,L:R] - v
               f[L:R,L:R] = f[L:R,L:R] - v
          if show: 
               pl.imshow(solution); pl.show()

##
## Identification of sources
##

j = 0
v_t = np.zeros((N_grid_omega,N_grid_omega,T_grid))
time = 1

while j < 2: 
     
     v_t = PDE.velocity(v,v_t,k)
     mask = np.zeros((N_grid_omega,N_grid_omega), dtype=bool)

     (time,mask) = PDE.support_mask(time,mask,v,v_t,zero_tol,nonzero_tol)
     print "Max at",time
     g[L:R,L:R] = np.multiply(v_t[:,:,time-1],mask) #fortran vs. python indexing
     print "Displaying Source"
     pl.imshow(g)
     pl.show()
     pl.imshow(np.multiply(v[:,:,time-1],mask))
     pl.colorbar()
     pl.show()

     temp = np.zeros((N_grid_omega,N_grid_omega,T_grid))
     temp = PDE.forwardmovie(temp,zeros_Rn,g,c,h,k)
     w = np.zeros((N_grid_omega,N_grid_omega,T_grid))
     w = PDE.oddreflect(w,temp,time)
     temp = None

     v = v - w
     v_t = PDE.velocity(v,v_t,k)

     print "Showing effect at true source times:"
     for t_j in t: #range(1,100):     
          print "Displacement"
          pl.imshow(np.multiply(v_t[:,:,t_j-1],mask))
          pl.colorbar()
          pl.show()
          
          pl.imshow(np.multiply(v_t[:,:,t_j],mask))
          pl.colorbar()
          pl.show()
          
          pl.imshow(np.multiply(v_t[:,:,t_j+1],mask))
          pl.colorbar()
          pl.show()
          print "Velocity",t_j
          pl.imshow(v_t[:,:,t_j-1])
          pl.colorbar()
          pl.show()

          pl.imshow(v_t[:,:,t_j])
          pl.colorbar()
          pl.show()
          
          pl.imshow(v_t[:,:,t_j+1])
          pl.colorbar()
          pl.show()


     if movie:     
          print "Movie of new waveform (source subtracted)"
          fig = pl.figure()
          if write: 
               for i in range(1,T_grid):
                    pl.imshow(v_t[:,:,i])
                    filename = "image/frame{0}.png".format(i)
                    pl.savefig(filename)
          else:
               def animate(i):
                    return pl.imshow(v[:,:,i])
               anim = animation.FuncAnimation(fig, animate, frames=T_grid/3, interval=50)
               pl.show()

     j = j + 1



