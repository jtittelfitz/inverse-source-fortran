#f2py -c --fcompiler=gnu95 --f90flags=-ffree-form -m PDE PDE.f90

import sys
import PDE
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as pl
import sklearn.linear_model as lr
import matplotlib as mpl
import matplotlib.animation as anim
import Image

pl.rcParams['figure.figsize'] = 20, 10

scale = 2
movie = True; write = False; show = False

h = 0.0125*scale; k = 0.0025*scale
r_omega = 2.; r_Rn = 5.
N_grid_omega = 2*int(r_omega/h) + 1; N_grid_Rn = 2*int(r_Rn/h) + 1
r_Rn = h*(N_grid_Rn - 1)/2
T = 4; T_grid = int(T/k) + 1
L = (N_grid_Rn - N_grid_omega)/2
R = N_grid_Rn - (N_grid_Rn - N_grid_omega)/2

# define constant zero matrices for both grids
zeros_Rn = np.zeros((N_grid_Rn,N_grid_Rn)); zeros_omega = np.zeros((N_grid_omega,N_grid_omega))

# initialize grid
c = np.zeros((N_grid_Rn,N_grid_Rn)); boundary_mask = np.zeros((N_grid_omega,N_grid_omega))
(c,boundary_mask) = PDE.initialize(c,boundary_mask,h,r_Rn,r_omega)

##
## Data simulation: initial data or source problem -> boundary data
##
print "-------------------------"
print "Beginning data simulation"
Lambda = np.zeros((N_grid_omega,N_grid_omega,T_grid)); force = np.zeros((N_grid_Rn,N_grid_Rn,T_grid))

source_speed = 1.
print "True source speed: ",source_speed


## set times where path changes direction
#path_times = np.array([0,0.125,0.25])/k
#path_times = np.array([0,1,1.5,2])/k     
path_times = (1/source_speed)*np.array([0.25,0.75,1.25])/k     
path_times = path_times.astype(int)

## set coordinates of points where path changes direction
#path_points = np.array([[-1, 0.,],[0.,0.],[0.,1],[0.5,0.5]])
path_points = np.array([[0.,-0.5],[0.,0.],[0.,0.5]])

if path_times.shape[0] <> path_points.shape[0]:
     print "Bad input; exiting"
     sys.exit()

## construct piecewise-linear, constant speed path between chosen times/points
path = np.zeros((path_times[path_times.size-1]+1,2))

for j in range(path_times.size - 1):
     for n in range(path_times[j],path_times[j+1]+1):     

          path[n,0] = (path_points[j,0] 
               + (path_points[j+1,0] - path_points[j,0])*(n - path_times[j])/(path_times[j+1] - path_times[j]))

          path[n,1] = (path_points[j,1] 
               + (path_points[j+1,1] - path_points[j,1])*(n - path_times[j])/(path_times[j+1] - path_times[j]))
                   
## construct forcing term: force is characteristic function of circle with given radius, moving along specified path
radius = 0.1
force = PDE.initializeforce(force,path,path_times[0]+1,radius,h,r_Rn)                    

force_O = force[L:R,L:R,:]

## generate fullwaveform from forcing term, measure solution on boundary
u = np.zeros((N_grid_omega,N_grid_omega,T_grid))
u = PDE.forwardsourcemovie(u,zeros_Rn,zeros_Rn,force,c,h,k)
 
## solve elliptic boundary value problem 
phi = np.zeros((N_grid_omega,N_grid_omega))
phi = PDE.elliptic(phi,u,c,h,boundary_mask)

## uncomment to show elliptic solution
# pl.imshow(phi); pl.show()

## solve reverse-time boundary value problem
v = np.zeros((N_grid_omega,N_grid_omega,T_grid))
v = PDE.reversemovie(v,u,phi,zeros_omega,c,h,k,boundary_mask)

## the rest of this is just experimentation/exploration to see what might characterize points/times where path changes direction

v_t = np.zeros((N_grid_omega,N_grid_omega,T_grid))
v_t = PDE.velocity(v,v_t,k)

u_t = np.zeros((N_grid_omega,N_grid_omega,T_grid))
u_t = PDE.velocity(u,u_t,k)

del_vt = np.zeros((N_grid_omega,N_grid_omega,T_grid))
del_vt = PDE.gradient(v_t,del_vt,h)

## find points of max velocity

tol = 0.1

max_u = np.zeros((N_grid_omega,N_grid_omega,T_grid))
max_locs_u = np.zeros((2,T_grid))
[max_u,max_locs_u] = PDE.maxmassmask(u_t,max_u,max_locs_u,r_omega,h,tol)
r_values_u = np.zeros((T_grid,T_grid))

max_v = np.zeros((N_grid_omega,N_grid_omega,T_grid))
max_locs_v = np.zeros((2,T_grid))
[max_v,max_locs_v] = PDE.maxmassmask(v_t,max_v,max_locs_v,r_omega,h,tol)
#print max_locs_v
r_values_v = np.zeros((T_grid,T_grid))


## linear regression on points of max velocity
start_index = 1
min_offset = 50
min_index = 200

for i in range(start_index,T_grid/2 - min_offset):
     for j in range(i + min_offset,T_grid/2):
          slope, intercept, r_value, p_value, std_err = stats.linregress(max_locs_u[:,i:j])     
          r_values_u[i,j] = r_value**2
          
          slope, intercept, r_value, p_value, std_err = stats.linregress(max_locs_v[:,i:j])     
          r_values_v[i,j] = r_value**2 

## best regression for true solution

#i = r_values_u.argmax()
i,j = np.unravel_index(r_values_u.argmax(), r_values_u.shape)
slope, intercept, r_value, p_value, std_err = stats.linregress(max_locs_u[:,i:j])
print "Start time, final time, slope, intercept:",i*k,j*k,slope,intercept

## estimate velocity for true solution

vel = 0

for l in range(i,j):
     #print j
     vel = vel + np.sqrt((max_locs_v[0,l] - max_locs_v[0,l+1])**2 + (max_locs_v[1,l] - max_locs_v[1,l+1])**2)/k
     
vel = vel/(j - i)
print "estimated source velocity: ", vel

## output points for true solution
dotplot = True

if dotplot:
     fig = pl.figure(0)
     ax1 = fig.add_subplot(2,2,1, title='Points of max velocity for true solution')
     ax2 = fig.add_subplot(2,2,2, title='Closeup and line of best fit')
     ax3 = fig.add_subplot(2,2,3, title='Points of max velocity for time-reversed solution')
     ax4 = fig.add_subplot(2,2,4, title='Closeup and line of best fit')

     ax1.plot(max_locs_u[0,0:T_grid/2],-max_locs_u[1,0:T_grid/2],'o')

xi = max_locs_u[0,i:j]
y = max_locs_u[1,i:j]
line = slope*xi + intercept
if dotplot: 
     ax2.plot(xi,-line,'r-',xi,-y,'o')


## best regression for time-reversed solution

#i = r_values_v.argmax()
i,j = np.unravel_index(r_values_v.argmax(), r_values_v.shape)
slope, intercept, r_value, p_value, std_err = stats.linregress(max_locs_v[:,i:j])
print "Start time, final time, slope, intercept:",i*k,j*k,slope,intercept

## estimate velocity for time-reversed solution

vel = 0

for l in range(i,j):
     #print vel, max_locs_v[0,j], max_locs_v[1,j]
     vel = vel + np.sqrt((max_locs_v[0,l] - max_locs_v[0,l+1])**2 + (max_locs_v[1,l] - max_locs_v[1,l+1])**2)/k
     
vel = vel/(j - i)

print "estimated source velocity: ", vel

## output points for time-reversed solution

if dotplot: 
     ax3.plot(max_locs_v[0,0:T_grid/2],-max_locs_v[1,0:T_grid/2],'o')

xi = max_locs_v[0,i:j]
y = max_locs_v[1,i:j]
line = slope*xi + intercept
if dotplot: 
     ax4.plot(xi,-line,'r-',xi,-y,'o')
     pl.show()

## guess at finding source

v_x = (xi[j-i-1]-xi[0])/((j-i)*k)

forward = 0
backward = 0

errors = np.zeros((20))

step = 1

for backward in range(0,10,step):

     path_times = (1/source_speed)*np.array([(i-backward)*k,(j + forward)*k])/k     
     path_times = path_times.astype(int)

     start_x = xi[0] - v_x*backward*k
     start_y = slope*start_x + intercept

     final_x = xi[j-i-1] + v_x*forward*k
     final_y = slope*final_x + intercept

     #path_points = np.array([[max_locs_v[0,i],max_locs_v[1,i]],[max_locs_v[0,j],max_locs_v[1,j]]])
     path_points = np.array([[start_y,start_x],[final_y,final_x]])

     print 'velocity',np.sqrt((xi[j-i-1]-xi[0])**2 + (line[j-i-1] - line[0])**2)/((j-i)*k)

     path = np.zeros((path_times[path_times.size-1]+1,2))

     for l in range(path_times.size - 1):
          for n in range(path_times[l],path_times[l+1]+1):     

               path[n,0] = (path_points[l,0] 
                    + (path_points[l+1,0] - path_points[l,0])*(n - path_times[l])/(path_times[l+1] - path_times[l]))

               path[n,1] = (path_points[l,1] 
                    + (path_points[l+1,1] - path_points[l,1])*(n - path_times[l])/(path_times[l+1] - path_times[l]))
                        
     ## construct forcing term: force is characteristic function of circle with given radius, moving along specified path
     radius = 0.1
     force = np.zeros((N_grid_Rn,N_grid_Rn,T_grid))
     force = PDE.initializeforce(force,path,path_times[0]+1,radius,h,r_Rn)                    

     force_O = force[L:R,L:R,:]

     ## generate fullwaveform from forcing term, measure solution on boundary
     u_guess = np.zeros((N_grid_omega,N_grid_omega,T_grid))
     u_guess = PDE.forwardsourcemovie(u_guess,zeros_Rn,zeros_Rn,force,c,h,k)

     ## compute error
     l2_error = np.zeros((T_grid))
     sup_error = np.zeros((T_grid))

     l2_error = PDE.l2error(l2_error,u,u_guess)
     sup_error = PDE.superror(sup_error,u,u_guess)
     temp  = 0
     temp = PDE.boundaryerror(temp,u,u_guess,boundary_mask)
     errors[backward:backward+step] = np.log(temp)

fig = pl.figure(0)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(l2_error)
ax2.plot(errors)
pl.show()





## movie of wave evolution

zero_v = np.zeros((N_grid_omega,N_grid_omega,T_grid))
#zero_v = PDE.zeromask(del_vt,zero_v)

#fig,ax = pl.subplots()
#cax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
#pl.clf()
#ims = []
#im = ax.imshow(u_guess[:,:,100] - u[:,:,100])

#ims.append([im])

#cax,kw = mpl.colorbar.make_axes([ax])

#for time in range(60,224):
 #    im = ax.imshow(np.random.random((10,10)))#u_guess[:,:,time] - u[:,:,time])  
  #   pl.colorbar(im, cax=cax, **kw)
     #pl.show()   
   #  ims.append([im])
     
#ani = anim.ArtistAnimation(fig,ims,interval=50,blit=False)  
#pl.show()

if movie:     
     print "Movie of time-reversed waveform"
     fig = pl.figure(1)
     ax1 = fig.add_subplot(2,3,1, title='True solution displacement')
     ax2 = fig.add_subplot(2,3,4, title='Time-reversed displacement')
     ax3 = fig.add_subplot(2,3,3, title='True solution points of max velocity')
     ax4 = fig.add_subplot(2,3,6, title='Time-reversed points of max velocity')
     ax5 = fig.add_subplot(2,3,2, title='True solution velocity')
     ax6 = fig.add_subplot(2,3,5, title='Time-reversed velocity')
     ims = []
     maxpath = np.zeros((N_grid_omega,N_grid_omega))
     maxpath_u = np.zeros((N_grid_omega,N_grid_omega))
     im5 = ax5.imshow(u[:,:,100] - u_guess[:,:,100])#u_t[:,:,time])
     fig.colorbar(im5)
     for time in range(T_grid/2):
          im  = ax1.imshow(u[:,:,time]) # true waveform
          im2 = ax2.imshow(u_guess[:,:,time]) #v[:,:,time]) # time-reversed from boundary waveform
          maxpath = maxpath + max_v[:,:,time]
          maxpath_u = maxpath_u + max_u[:,:,time]
          im3, = ax3.plot(max_locs_u[0,0:time],-max_locs_u[1,0:time],'o')#imshow(maxpath_u)#6*max_v[:,:,time] - 3*force_O[:,:,time])#del_vt[:,:,time])#
          im4, = ax4.plot(max_locs_v[0,0:time],-max_locs_v[1,0:time],'o')#imshow(maxpath)
          im5 = ax5.imshow(u[:,:,time] - u_guess[:,:,time])#u_t[:,:,time])
          im6 = ax6.imshow(v_t[:,:,time])
          
          #im3 = ax3.imshow((u[:,:,time] - v[:,:,time])/u[:,:,time].max())
          #im.set_clim(-300,300)  
          #im2.set_clim(-300,300)
          #im3.set_clim(-3,9)          
          ims.append([im,im2,im3,im4,im5,im6])
          if write:           
               strtime = str(time).zfill(4)
               filename = "image6/frame{0}.png".format(strtime)
               pl.savefig(filename)

     #def animate(i):
     #     return pl.imshow(v[:,:,3*i])
     #anim = animation.FuncAnimation(fig, animate, frames=T_grid/3, interval=50)
     ani = anim.ArtistAnimation(fig,ims,interval=50,blit=False)  

     pl.show()

