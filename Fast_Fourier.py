# COP_Project_5_2

import numpy as np
import matplotlib.pyplot as plt

nx, ny     = (1000, 1000)        # number of points in x and y directions
xmax, ymax = 50, 50
x          = np.linspace(-xmax, xmax, nx)
y          = np.linspace(-ymax, ymax, ny)
X, Y       = np.meshgrid(x, y)
dx         = x[1] - x[0]
dy         = y[1] - y[0]
a		   = 2
Z          = (X*X+Y*Y)<a**2          # circular hole

ZFT        = np.fft.fftshift(np.fft.fft2(Z))  # compute 2D FFT and shift the zero-frequency component to the center of the spectrum.
kx         = (-nx/2 + np.arange(0,nx))*2*np.pi/(2*xmax)
ky         = (-ny/2 + np.arange(0,ny))*2*np.pi/(2*ymax)

phi  = np.linspace(0, 2*np.pi, 100)       # phi-grid for plotting a circle
kxc1 = (3.83171/a)*np.cos(phi)            # 1st zero of Airy function
kyc1 = (3.83171/a)*np.sin(phi)
kxc2 = (7.01559/a)*np.cos(phi)            # 2nd zero of Airy function
kyc2 = (7.01559/a)*np.sin(phi)

# plot Z and it's Fourier transform ZFT
fig, (ax0, ax1) = plt.subplots(ncols=2)
plot1 = ax0.pcolormesh(X,Y,Z)
KX, KY     = np.meshgrid(kx, ky)
plot2 = ax1.pcolormesh(KX,KY,np.abs(ZFT))
zero1 = ax1.plot(kxc1, kyc1, color = 'white')
zero2 = ax1.plot(kxc2, kyc2, color = 'white')
ax0.set_title('real space')
ax1.set_title('Fourier space')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_aspect('equal')
ax0.set_aspect('equal')
plt.show()

