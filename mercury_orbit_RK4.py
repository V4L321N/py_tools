import numpy as np
import matplotlib.pyplot as plt
################################################################################
def Kepler(y):
    gam = 4.0*(np.pi**2.0)
    dy0 = y[2]
    dy1 = y[3]
    r32 = (y[0]*y[0] + y[1]*y[1])**(1.5)
    dy2 = -gam*y[0]/r32
    dy3 = -gam*y[1]/r32 
    return np.array([dy0, dy1, dy2, dy3])
################################################################################
def RK4(f,t,dt,initial):
     gam   = 4.0*(np.pi**2.0)
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

     T  = 0.5*(y[2,:]*y[2,:] + y[3,:]*y[3,:]) # kinetic energy
     V  = -gam/np.sqrt((y[0,:]*y[0,:] + y[1,:]*y[1,:])) # potential energy
     E  = T + V   # total energy
     return tlist, y, E  
################################################################################
# MAIN PROGRAM 
tstart      = 0.0
tend        = 5.0
initial     = np.array( [1.0, 0.0, 0.0, 2.0*np.pi] )

t1, y1, E1  = RK4(Kepler,tend,0.10,initial)
t2, y2, E2  = RK4(Kepler,tend,0.05,initial)
t3, y3, E3  = RK4(Kepler,tend,0.01,initial)
t4, y4, E4  = RK4(Kepler,tend,0.001,initial)

#plot results
f, axarr = plt.subplots(2,1)

axarr[0].plot(y1[0,:],y1[1,:],'k-',label='RK4 $\Delta t$ = 0.10')
axarr[0].plot(y2[0,:],y2[1,:],'r-',label='RK4 $\Delta t$ = 0.05')
axarr[0].plot(y3[0,:],y3[1,:],'b-',label='RK4 $\Delta t$ = 0.01')
axarr[0].plot(y4[0,:],y4[1,:],'g--',label='RK4 $\Delta t$ = 0.001')
axarr[0].set_xlabel('$x$ (AU)')
axarr[0].set_ylabel('$y$ (AU)')
axarr[0].set_aspect('equal', 'datalim')
axarr[0].legend(loc='upper left', shadow=False, fontsize=12)

Eexact = -2.0*np.pi**2.0
dtlist = np.array([0.1, 0.05, 0.01, 0.001])
DeltaE = np.array([E1[len(E1)-1],E2[len(E2)-1],E3[len(E3)-1],E4[len(E4)-1]])
DeltaE = np.abs(DeltaE - Eexact)
axarr[1].loglog(dtlist,DeltaE,'k*-')
axarr[1].set_xlabel('$\\Delta t$ (years)')
axarr[1].set_ylabel('$E(t=5)-E_{exact}$ (AU)')

plt.savefig("exercise15.pdf")
plt.show()  



