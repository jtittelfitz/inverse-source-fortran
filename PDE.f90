! -*- f90 -*
!module PDE
!implicit none

!contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!	
! Main methods
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine initialize(c,boundary_mask,h,r_Rn,r_O,N_R,N_O)
     implicit none

     integer :: N_R, N_O
     double precision, dimension(N_R,N_R) :: c
     double precision, dimension(N_O,N_O) :: boundary_mask
     double precision :: h, r_Rn, R_O
!f2py intent(in) h, r_Rn, r_O
!f2py intent(in,out) c, boundary_mask
!f2py integer, intent(hide), depend(c) :: N_R = shape(c,0)
!f2py integer, intent(hide), depend(boundary_mask) :: N_O = shape(boundary_mask,0)
     real(8) :: x,y,x2,y2
     integer :: i,j
    
     boundary_mask = 0

     do i = 1,N_R
     x = -r_Rn + (i-1)*h; x2 = -r_O + (i-1)*h
     do j = 1,N_R
          y = -r_Rn + (j-1)*h; y2 = -r_O + (j-1)*h
	     !c(i,j) = 1
	     c(i,j) = 1 + 0.01*sin(x) + 0.01*cos(y)
	     !print *,c(i,j)	     
	     if ((x2**2 + y2**2 .ge. (2-2*h)**2) .and.((i .le. N_O) .and. (j .le. N_O))) then !.and. (x2**2 + y2**2 .le. (2+2*h)**2)) then !
               boundary_mask(i,j) = 1          
          endif
     end do
     end do
          
end subroutine

subroutine newspeed(c,h,r_Rn,r_O,N_R)
     implicit none

     integer :: N_R, N_O
     double precision, dimension(N_R,N_R) :: c
     double precision :: h, r_Rn, R_O
!f2py intent(in) h, r_Rn, r_O
!f2py intent(in,out) c
!f2py integer, intent(hide), depend(c) :: N_R = shape(c,0)
     real(8) :: x,y,x2,y2
     integer :: i,j

     do i = 1,N_R
     x = -r_Rn + (i-1)*h; x2 = -r_O + (i-1)*h
     do j = 1,N_R
          y = -r_Rn + (j-1)*h; y2 = -r_O + (j-1)*h
	     c(i,j) = 1
	     !c(i,j) = 2 + 0.1*sin(x) + 0.1*cos(y)
	     !print *,c(i,j)	     	     
     end do
     end do
          
end subroutine

subroutine circle(g,x_0,y_0,r,h,r_Rn,N_R)
     implicit none

     integer :: N_R
     double precision, dimension(N_R,N_R) :: g
     double precision :: x_0,y_0,r,h,r_Rn
!f2py intent(in) x_0,y_0,r,h, r_Rn, N_R
!f2py intent(in,out) g
!f2py integer, intent(hide), depend(g) :: N_R = shape(g,0)

     double precision :: x,y
     integer :: i,j

     do i = 1,N_R
     do j = 1,N_R
	     x = -r_Rn + (i-1)*h; y = -r_Rn + (j-1)*h
	     if ((x-x_0)**2 + (y-y_0)**2 .lt. r**2) then
               g(i,j) = 1              
          else
               g(i,j) = 0
          endif
     end do
     end do
end subroutine

subroutine initializeforce(force,path,start_time,radius,h,r_Rn,N_R,T,T_path)
     implicit none
     
     integer :: N_R,T,T_path,start_time
     double precision :: radius,h,r_Rn
     double precision, dimension(N_R,N_R,T) :: force
     double precision, dimension(T_path,2) :: path
!f2py intent(in) path,radius,h,r_Rn,start_time
!f2py intent(in,out) force
!f2py integer, intent(hide), depend(force) :: N_R = shape(force,0), T = shape(force,2)
!f2py integer, intent(hide), depend(path) :: T_path = shape(path,0)

     integer :: n,i,j
     double precision :: x_0, y_0,x,y
     
     do n = start_time,T_path
          x_0 = path(n,1); y_0 = path(n,2)
          do i = 1,N_R
               x = -r_Rn + (i-1)*h
               do j = 1,N_R
                    y = -r_Rn + (j-1)*h 
                    if ((x - x_0)**2 + (y - y_0)**2 < radius**2) then
                         force(i,j,n) = 1
                    else
                         force(i,j,n) = 0
                    end if                    
               end do
          end do
     end do
     
end subroutine

subroutine forwardmovie(u,f,g,c,h,k,N_R,N_O,T)
     implicit none

     integer :: N_R,N_O,T
     double precision :: h,k
     double precision, dimension(N_R,N_R) :: f,g,c
     double precision :: u(N_O,N_O,T)
     double precision, dimension(N_R,N_R) :: u1,u2,u3
!f2py intent(in,out) u
!f2py intent(in) f,g,c,h,k
!f2py integer, intent(hide), depend(f) :: N_R = shape(f,0)
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T = shape(u,2)

     integer :: n,i,j
     integer :: L, R  
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2

	print *,"Starting (full) finite diff. sim. of forward problem"
     u2 = f
     u(:,:,1) = u2(L:R,L:R)

     do i = 2,N_R-1
     do j = 2,N_R-1          
          u3(i,j) = u2(i,j) + k*g(i,j) + 0.5*(k**2)*c(i,j)*(u2(i+1,j)&
          & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2
     end do
     end do
	
     u1 = u2
     u2 = u3
     u(:,:,2) = u2(L:R,L:R)
	
     do n = 2,T-1
	     do i = 1,N_R
	     do j = 1,N_R
			u3(i,j) = 2*u2(i,j) - u1(i,j) + (k**2)*c(i,j)*(u2(i+1,j)&
              & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2
	     end do
	     end do
	     
	     !u3 = 2*u2 - u1 + k**2*c*(cshift(u2,1,DIM = 1) + cshift(u2,-1,DIM = 1) + cshift(u2,1,DIM = 2) + cshift(u2,-1,DIM = 2) - 4*u2)/h**2
	     u1 = u2
	     u2 = u3
          u(:,:,n+1) = u2(L:R,L:R)
     end do
	
     print *,"Done with (full) forward"
end subroutine

subroutine forwardsourcemovie(u,f,g,force,c,h,k,N_R,N_O,T)
     implicit none

     integer :: N_R,N_O,T
     double precision :: h,k
     double precision, dimension(N_R,N_R) :: f,g,c
     double precision :: u(N_O,N_O,T)
     double precision :: force(N_R,N_R,T)
     double precision, dimension(N_R,N_R) :: u1,u2,u3
!f2py intent(in,out) u
!f2py intent(in) f,g,force,c,h,k
!f2py integer, intent(hide), depend(f) :: N_R = shape(f,0)
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T = shape(u,2)

     integer :: n,i,j
     integer :: L, R  
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2

	print *,"Starting (full) finite diff. sim. of forward problem"
     u2 = f
     u(:,:,1) = u2(L:R,L:R)

     do i = 2,N_R-1
     do j = 2,N_R-1          
          u3(i,j) = u2(i,j) + k*g(i,j) + 0.5*(k**2)*c(i,j)*(u2(i+1,j)&
          & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2&
          & + force(i,j,1)
     end do
     end do
	
     u1 = u2
     u2 = u3
     u(:,:,2) = u2(L:R,L:R)
	
     do n = 2,T-1
	     do i = 1,N_R
	     do j = 1,N_R
			u3(i,j) = 2*u2(i,j) - u1(i,j) + (k**2)*c(i,j)*(u2(i+1,j)&
              & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2&
              & + force(i,j,n)
	     end do
	     end do
	     
	     !u3 = 2*u2 - u1 + k**2*c*(cshift(u2,1,DIM = 1) + cshift(u2,-1,DIM = 1) + cshift(u2,1,DIM = 2) + cshift(u2,-1,DIM = 2) - 4*u2)/h**2
	     u1 = u2
	     u2 = u3
          u(:,:,n+1) = u2(L:R,L:R)
     end do
	
     print *,"Done with (full) forward"
end subroutine

subroutine forward(Lambda,f,g,c,h,k,boundary_mask,N_R,N_O,T)
     implicit none

     integer :: N_R,N_O,T
     double precision :: h,k
     double precision, dimension(N_R,N_R) :: f,g,c
     double precision :: Lambda(N_O,N_O,T)
     double precision :: boundary_mask(N_O,N_O)
     double precision, dimension(N_R,N_R) :: u1,u2,u3
!f2py intent(in,out) Lambda
!f2py intent(in) f,g,c,h,k, boundary_mask
!f2py integer, intent(hide), depend(f) :: N_R = shape(f,0)
!f2py integer, intent(hide), depend(Lambda) :: N_O = shape(Lambda,1), T = shape(Lambda,2)

     double precision :: temp(N_O,N_O)
     double precision :: diff(N_R,N_R)
     integer :: n,i,j
     integer :: L, R  
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2
     
     diff = 0

	print *,"Starting finite diff. sim. of forward problem"
     u2 = f
     temp = u2(L:R,L:R)
     where (boundary_mask == 1)
          Lambda(:,:,1) = temp
     end where

     do i = 2,N_R-1
     do j = 2,N_R-1          
          u3(i,j) = u2(i,j) + k*g(i,j) + 0.5*(k**2)*c(i,j)*(u2(i+1,j)&
          & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2
     end do
     end do
	
     u1 = u2
     u2 = u3
     temp = u2(L:R,L:R)
     where (boundary_mask == 1)
          Lambda(:,:,2) = temp
     end where
	
     do n = 2,T-1
	     do i = 1,N_R
	     do j = 1,N_R
			u3(i,j) = 2*u2(i,j) - u1(i,j) + (k**2)*c(i,j)*(u2(i+1,j)&
               & + u2(i-1,j) + u2(i,j+1) + u2(i,j-1) - 4*u2(i,j))/h**2
	     end do
	     end do
	     
	     !u3 = 2*u2 - u1 + k**2*c*(&
	     ! &(-1/12)*cshift(u2,-2,DIM = 1)+&
	     ! &(4/3)*cshift(u2,-1,DIM = 1)+&
	     ! &(4/3)*cshift(u2,1,DIM = 1)+&
	     ! &(-1/12)*cshift(u2,2,DIM = 1)+&
	     ! &(-1/12)*cshift(u2,-2,DIM = 2)+&
	     ! &(4/3)*cshift(u2,-1,DIM = 2)+&
	     ! &(4/3)*cshift(u2,1,DIM = 2)+&
	     ! &(-1/12)*cshift(u2,2,DIM = 2)+&
	     ! &(-5)*u2)/h**2	     	     
	     
	     u1 = u2
	     u2 = u3
          temp = u2(L:R,L:R)
          where (boundary_mask == 1)
               Lambda(:,:,n+1) = temp
          end where
     end do
	
     print *,"Done with forward"
end subroutine


subroutine reverse(v,Lambda,f,g,c,h,k,boundary_mask,N_R,N_O,T)
     implicit none

     integer :: N_R,N_O,T
     double precision :: h,k
     double precision, dimension(N_R,N_R) :: c
     double precision :: Lambda(N_O,N_O,T)
     double precision, dimension(N_O,N_O) :: f,g,v,v1,v2,v3,boundary_mask
!f2py intent(in) Lambda,f,g,c,h,k,boundary_mask
!f2py intent(in,out) v
!f2py integer, intent(hide), depend(c) :: N_R = shape(c,0)
!f2py integer, intent(hide), depend(Lambda) :: N_O = shape(Lambda,1), T = shape(Lambda,2)

     integer :: n,i,j
     integer :: L, R
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2

	print *,"Starting finite diff. sim. of reverse problem"
     v2 = f
     where (boundary_mask == 1)
          v2 = Lambda(:,:,T)
     end where

     do i = 2,N_O-1
     do j = 2,N_O-1          
     ! c is defined on larger grid; thus indexing in c is offset
          v3(i,j) = v2(i,j) + k*g(i,j) + 0.5*(k**2)*c(L - 1 + i,L - 1 + j)*(v2(i+1,j)&
          & + v2(i-1,j) + v2(i,j+1) + v2(i,j-1) - 4*v2(i,j))/h**2
     end do
     end do
     where (boundary_mask == 1)
          v3 = Lambda(:,:,T-1)
     end where
     
     v1 = v2
     v2 = v3     	
     do n = 2,T-1
	     do i = 1,N_O
	     do j = 1,N_O
			v3(i,j) = 2*v2(i,j) - v1(i,j) + (k**2)*c(L - 1 + i,L - 1 + j)*(v2(i+1,j)&
               & + v2(i-1,j) + v2(i,j+1) + v2(i,j-1) - 4*v2(i,j))/h**2
	     end do
	     end do
          where (boundary_mask == 1)
               v3 = Lambda(:,:,T-n)
          end where
	          
	     v1 = v2
	     v2 = v3          
     end do
	
	v = v3
	
     print *,"Done with reverse!"
end subroutine

subroutine elliptic(phi,Lambda,c,h,boundary_mask,N_R,N_O,T)	
	implicit none
	
	integer :: N_R,N_O,T
	double precision :: h
	double precision :: phi(N_O,N_O), boundary_mask(N_O,N_O)
	double precision :: Lambda(N_O,N_O,T)
	double precision :: c(N_R,N_R)
!f2py intent(in) Lambda, c,h,boundary_mask
!f2py intent(in,out) phi
!f2py integer, intent(hide), depend(c) :: N_R = shape(c,0)
!f2py integer, intent(hide), depend(Lambda) :: N_O = shape(Lambda,1), T = shape(Lambda,2)

     integer :: N_iter,n,i,j
     integer :: L, R
     N_iter = 1000 !number of times to iterate Gauss-Seidel     
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2
     
     where(boundary_mask == 1)
          phi = Lambda(:,:,T)
     end where
          
     do n = 1,N_iter
          do i = 2,N_O - 1
          do j = 2,N_O - 1
               if (boundary_mask(i,j) == 0) then
                    phi(i,j) = 0.25*(phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1))
               endif
          end do 
          end do
     end do
     
	print *,"elliptic"
end subroutine elliptic

subroutine reversemovie(v,Lambda,f,g,c,h,k,boundary_mask,N_R,N_O,T)
     implicit none

     integer :: N_R,N_O,T
     double precision :: h,k
     double precision, dimension(N_R,N_R) :: c
     double precision :: Lambda(N_O,N_O,T)
     double precision, dimension(N_O,N_O) :: f,g,v1,v2,v3,boundary_mask,c_O
     double precision, dimension(N_O,N_O,T) :: v
!f2py intent(in) Lambda,f,g,c,h,k,boundary_mask
!f2py intent(in,out) v
!f2py integer, intent(hide), depend(c) :: N_R = shape(c,0)
!f2py integer, intent(hide), depend(Lambda) :: N_O = shape(Lambda,1), T = shape(Lambda,2)
     
     integer :: n,i,j
     integer :: L, R
     L = (N_R - N_O)/2 + 1
     R = N_R - (N_R - N_O)/2
     
     c_O = c(L:R,L:R)

	print *,"Starting finite diff. sim. of reverse problem"
     v2 = f
     where (boundary_mask == 1)
          v2 = Lambda(:,:,T)
     end where
     
     v(:,:,T) = v2

     do i = 2,N_O-1
     do j = 2,N_O-1          
     ! c is defined on larger grid; thus indexing in c is offset
          v3(i,j) = v2(i,j) + k*g(i,j) + 0.5*(k**2)*c(L - 1 + i,L - 1 + j)*(v2(i+1,j)&
          & + v2(i-1,j) + v2(i,j+1) + v2(i,j-1) - 4*v2(i,j))/h**2
     end do
     end do
     where (boundary_mask == 1)
          v3 = Lambda(:,:,T-1)
     end where 
     
     v(:,:,T-1) = v3
     
     v1 = v2
     v2 = v3     	
     do n = 2,T-1
	     do i = 1,N_O
	     do j = 1,N_O
			v3(i,j) = 2*v2(i,j) - v1(i,j) + (k**2)*c(L - 1 + i,L - 1 + j)*(v2(i+1,j)&
               & + v2(i-1,j) + v2(i,j+1) + v2(i,j-1) - 4*v2(i,j))/h**2
	     end do
	     end do
	     
	     !v3 = 2*v2 - v1 + k**2*c_O*(&
	     ! &(-1/12)*cshift(v2,-2,DIM = 1)+&
	     ! &(4/3)*cshift(v2,-1,DIM = 1)+&
	     ! &(4/3)*cshift(v2,1,DIM = 1)+&
	     ! &(-1/12)*cshift(v2,2,DIM = 1)+&
	     ! &(-1/12)*cshift(v2,-2,DIM = 2)+&
	     ! &(4/3)*cshift(v2,-1,DIM = 2)+&
	     ! &(4/3)*cshift(v2,1,DIM = 2)+&
	     ! &(-1/12)*cshift(v2,2,DIM = 2)+&
	     ! &(-5)*v2)/h**2
	     
	     
          where (boundary_mask == 1)
               v3 = Lambda(:,:,T-n)
          end where 
	     
	     v(:,:,T - n) = v3
	     
	     v1 = v2
	     v2 = v3          
     end do
		
     print *,"Done with reverse"
end subroutine

subroutine velocity(u,u_t,k,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision :: k
     double precision, dimension(N_O,N_O,T) :: u, u_t
!f2py intent(in) u
!f2py intent(in,out) u_t
!f2py intent(in) k
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)
     integer :: n

     do n = 2,T-1
          u_t(:,:,n) = (u(:,:,n+1) - u(:,:,n-1))/(2*k)
     end do
     u_t(:,:,1) = u_t(:,:,2)
     u_t(:,:,T) = u_t(:,:,T)
     
end subroutine velocity

subroutine gradient (u,del_u,h,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision :: h
     double precision, dimension(N_O,N_O,T) :: u, del_u
!f2py intent(in) u
!f2py intent(in,out) del_u
!f2py intent(in) h
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)
     integer :: n,i,j

     do n = 1,T
          do i = 1,N_O
          do j = 1,N_O
               del_u(i,j,n) = 0.5*sqrt((u(i+1,j,n) - u(i-1,j,n-1))**2 + (u(i,j+1,n) - u(i,j-1,n-1))**2)/h
          end do
          end do
     end do
          
end subroutine gradient

subroutine maxmask(u,max_u,max_locs,r_O,h,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u, max_u
     double precision, dimension(2,T) :: max_locs
     double precision :: r_O,h
!f2py intent(in) u,r_O,h
!f2py intent(in,out) max_u, max_locs
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n,i,j,num_points
     double precision :: M,tol,x,y,xbar,ybar
          
     integer, dimension(2) :: max_xy
     
     tol = 0.1
    
     do n = 1,T
          M = maxval(u(:,:,n))
          
          where (ABS(u(:,:,n) - M) < tol*M)
               max_u(:,:,n) = u(:,:,n)/M
          elsewhere
               max_u(:,:,n) = 0
          end where
     end do
         
     !single point which is max-value
     do n = 1,T          
          max_xy = maxloc(u(:,:,n))
          
          x = -r_O + (max_xy(2) - 1)*h; y = -r_O + (max_xy(1) - 1)*h
          ! rows/columns vs. x/y
          max_locs(1,n) = x; max_locs(2,n) = y
     end do               
end subroutine maxmask

subroutine maxmassmask(u,max_u,max_locs,r_O,h,tol,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u, max_u
     double precision, dimension(2,T) :: max_locs
     double precision :: r_O,h,tol
!f2py intent(in) u,r_O,h,tol
!f2py intent(in,out) max_u, max_locs
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n,i,j,num_points
     double precision :: M,x,y,xbar,ybar,mass
          
     integer, dimension(2) :: max_xy
         
     do n = 1,T
          M = maxval(u(:,:,n))
          
          where (ABS(u(:,:,n) - M) < tol*M)
               max_u(:,:,n) = u(:,:,n)/M
          elsewhere
               max_u(:,:,n) = 0
          end where
     end do
     
     ! center of mass
     do n = 1,T
          mass = 0; num_points = 0
          xbar = 0; ybar = 0
          M = maxval(u(:,:,n))
          !if (n < 2) then
               !print *,M
          !end if
          do i = 1,N_O
          do j = 1,N_O
               if ((ABS(u(i,j,n) - M) < tol*M) .or. (M < tol)) then
                    num_points = num_points + 1
                    !if (n < 26) then 
                    !     print *,n,num_points
                    !end if
                    mass = mass + u(i,j,n)
                    !xbar = xbar + u(i,j,n)*(-r_O + (j - 1)*h); ybar = ybar + u(i,j,n)*(-r_O + (i - 1)*h)
                    xbar = xbar + (-r_O + (j - 1)*h); ybar = ybar + (-r_O + (i - 1)*h)
               end if               
          end do
          end do
          !max_locs(1,n) = xbar/mass; max_locs(2,n) = ybar/mass
          max_locs(1,n) = xbar/num_points; max_locs(2,n) = ybar/num_points          
     end do
     
     ! single point which is max-value
     !do n = 1,T          
     !     max_xy = maxloc(u(:,:,n))
     !     
     !     x = -r_O + (max_xy(2) - 1)*h; y = -r_O + (max_xy(1) - 1)*h
     !     ! rows/columns vs. x/y
     !     max_locs(1,n) = x; max_locs(2,n) = y
     !end do

               
end subroutine maxmassmask

subroutine zeromask(u,zero_u,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u, zero_u
!f2py intent(in) u
!f2py intent(in,out) zero_u
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n
     double precision :: tol
     
     tol = 0.1
     
     do n = 1,T
          
          !print *,M
          
          where (ABS(u(:,:,n)) < tol)
               zero_u(:,:,n) = 1
          elsewhere
               zero_u(:,:,n) = 0
          end where
     end do
               
end subroutine zeromask

subroutine support_mask(time,mask,u,u_t,zero_tol,nonzero_tol,N_O,T)
     implicit none
     
     integer :: N_O,T,time
     double precision :: zero_tol,nonzero_tol
     logical, dimension(N_O,N_O) :: mask
     double precision, dimension(N_O,N_O,T) :: u, u_t
!f2py intent(in) u,u_t
!f2py intent(in,out) mask,time
!f2py intent(in) t_1,zero_tol,nonzero_tol
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T = shape(u,2)
     logical :: temp_mask(N_O,N_O)
     integer n,c,current_max
     current_max = 0
     do n = 1,T
          where (( u(:,:,n)**2 < zero_tol) .and. (u_t(:,:,n)**2 > nonzero_tol))
               temp_mask = .true.
          elsewhere
               temp_mask = .false.
          end where 
          c = count(temp_mask) 
          if (c > current_max) then
               print *,c,n
               current_max = c
               time = n
               mask = temp_mask
          endif
          !if (n < 100) then 
               !print *,c,n,current_max
          !endif
     end do          
          
end subroutine support_mask

subroutine oddreflect(w,u,t_0,N_O,T)
     implicit none
     
     integer :: N_O,T,t_0
     double precision, dimension(N_O,N_O,T) :: u,w
!f2py intent(in) u,t_0
!f2py intent(in,out) w
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T = shape(u,2)
     w(:,:,t_0:T) = u(:,:,1:T-t_0+1)
     w(:,:,1:t_0-1) = -u(:,:,t_0:2:-1)

	print *,"oddreflect"
end subroutine oddreflect

subroutine diff_scheme(diff,u,k,h)
     double precision :: k,h
     double precision, dimension(:,:), intent(in) :: u
     double precision, dimension(:,:), intent(inout) :: diff
     
     diff = k**2*(cshift(u,1,DIM = 1) + cshift(u,-1,DIM = 1) &
      & + cshift(u,1,DIM = 2) + cshift(u,-1,DIM = 2) - 4*u)/h**2

!	print *,"diff_scheme"
end subroutine diff_scheme

subroutine l2norm(l2_norm,u,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u
     double precision, dimension(T) :: l2_norm
!f2py intent(in) u
!f2py intent(in,out) l2_norm
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n,i,j
     double precision :: numerator
     
     do n = 1,T
          numerator = 0
          do i = 1,N_O
               do j = 1,N_O
                    numerator = numerator + (u(i,j,n))**2                  
               end do
          end do
          l2_norm(n) = sqrt(numerator)
     end do
               
end subroutine l2norm

subroutine superror(sup_error,u,u_guess,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u, u_guess
     double precision, dimension(T) :: sup_error
!f2py intent(in) u, u_guess
!f2py intent(in,out) sup_error
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n
     
     do n = 1,T
          sup_error(n) = maxval(ABS(u(:,:,n) - u_guess(:,:,n)))/maxval(ABS(u(:,:,n)))
     end do
               
end subroutine superror

subroutine l2error(l2_error,u,u_guess,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O,T) :: u, u_guess
     double precision, dimension(T) :: l2_error
!f2py intent(in) u, u_guess
!f2py intent(in,out) l2_error
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n,i,j
     double precision :: numerator, denominator
     
     do n = 1,T
          numerator = 0; denominator = 1
          do i = 1,N_O
               do j = 1,N_O
                    numerator = numerator + (u(i,j,n) - u_guess(i,j,n))**2
                    denominator = denominator + u(i,j,n)**2
               end do
          end do
          l2_error(n) = sqrt(numerator/denominator)
     end do
               
end subroutine l2error

subroutine boundaryerror(bound_error,u,u_guess,boundary_mask,N_O,T)
     implicit none
     
     integer :: N_O,T
     double precision, dimension(N_O,N_O) :: boundary_mask
     double precision, dimension(N_O,N_O,T) :: u, u_guess
     double precision :: bound_error
!f2py intent(in) u, u_guess, boundary_mask
!f2py intent(in,out) bound_error
!f2py integer, intent(hide), depend(u) :: N_O = shape(u,0), T= shape(u,2)

     integer :: n,i,j
     double precision :: numerator, denominator

     numerator = 0; denominator = 1
     
     do n = 1,T
          do i = 1,N_O
               do j = 1,N_O
                    numerator = numerator + (boundary_mask(i,j)*(u(i,j,n) - u_guess(i,j,n)))**2
                    denominator = denominator + (boundary_mask(i,j)*u(i,j,n))**2
               end do
          end do
     end do
     bound_error = sqrt(numerator/denominator)
               
end subroutine boundaryerror

!end module

