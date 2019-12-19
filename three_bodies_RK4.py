
import numpy as np
import matplotlib.pyplot as plt

gam = 4.0*(np.pi**2.0)
M1  = 0.0507/330000    # mass of mercury relative sun
M2  = 1*0.8196/330000  # mass of venus relative to sun

def Three_Bodies(y):        # definition of differential equation system
    dy    = np.zeros(len(y),float)	# generating empty array
    dy[0] = y[4]
    dy[1] = y[5]
    dy[2] = y[6]
    dy[3] = y[7]
    r1    = (y[0]*y[0] + y[1]*y[1])**(1.5)   # distance of mercury from sun
    r2    = (y[2]*y[2] + y[3]*y[3])**(1.5)   # distance of venus from sun
    r12   = ((y[0]-y[2])*(y[0]-y[2]) + (y[1]-y[3])*(y[1]-y[3]))**(1.5)  # distance between planets
    f12x  = gam*(y[0]-y[2])/r12	# constructing the equations (1) - (4) and filling up the 'dy' array
    f12y  = gam*(y[1]-y[3])/r12
    dy[4] = -gam*y[0]/r1 - M2*f12x
    dy[5] = -gam*y[1]/r1 - M2*f12y
    dy[6] = -gam*y[2]/r2 + M1*f12x
    dy[7] = -gam*y[3]/r2 + M1*f12y    
    return np.array(dy)

def RK4(f,t,dt,initial):          # Runge-Kutta method as taken from exercise15
     m     = initial.shape[0]
     tlist = np.arange(0,t+dt,dt)
     n     = len(tlist)
     y     = np.zeros((m,n),float)
     y[:,0]= initial
     for i in range(0,n-1):
         k1       = f(y[:,i])                              # Eq. (5.25)
         k2       = f(y[:,i] + 0.5*dt*k1)                  # Eq. (5.26)
         k3       = f(y[:,i] + 0.5*dt*k2)                  # Eq. (5.27)
         k4       = f(y[:,i] +     dt*k3)                  # Eq. (5.28)
         y[:,i+1] = y[:,i] + (dt/6.0)*(k1 +2*k2+2*k3 +k4)  # Eq. (5.24)
     return tlist, y

# run exercise 6.1
tstart      = 0.0
tend        = 110.0
initial   = np.array( [0.3075, 0.0, 0.723, 0.0, 0.0, 12.45, 0.0, 7.3986] )
t, y  = RK4(Three_Bodies,tend,0.0005,initial)

#plot results

plt.plot(y[0,:],y[1,:],'bo',label='mercury')
plt.plot(y[2,:],y[3,:],'ro',label='venus')
plt.plot([0],[0],'yo',label='sun')
plt.legend(loc='upper left', shadow=False, fontsize=12)

plt.show()  




