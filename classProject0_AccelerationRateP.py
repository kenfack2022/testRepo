# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:13:56 2020
@author: Anatole Kenfack
This code successfully reproduces results 
(Current and Energy) published
in PRL94, 110603 (2005) & PRL100, 044104 (2008)
"""
import numpy as np
import matplotlib.pyplot as plt

nx = 8*512
mmax = 100
alp = 0.3
iu = complex(0,1)
#print(np.pi)
dt = 1.
# Space wavelength L = 2.*pi 
L = 2.*np.pi
# Extreme points of the spatial coordinates
xmin = -L/2
xmax = L/2

# Step site dx = x_i-x_(i-1), x0=xmin, x1= xmin +dx, x2 = xmin +2dx,
# x4 = xmin + 3dx .......... xmax = xmin + (N-1)dx ---> dx = (xmax-xmin)/N
dx = L/nx

# kbar is the effective Planck constant
kbar = 3.3750*np.pi
# kicking strenght
P = 0.5
# the potential energy function
x = np.zeros(nx, float)
Vratchet = np.zeros(nx, float)
psi = np.zeros(nx, complex)
densPsi = np.zeros(nx, float)
kx = np.zeros(nx, float)
T = np.zeros(nx, float)
# variables containing 1st and 2nd derivatives of psi
# <k> and <k^2>
d1Psi = np.zeros(nx, complex)
d2Psi = np.zeros(nx, complex)
# implemeting and visualizing the ratchet potential
x = np.linspace(xmin,xmax,nx)
Vratchet = np.sin(x) + alp*np.sin(2.*x)

plt.xlabel('position x')
plt.ylabel('V(x)')
plt.plot(x,Vratchet)
plt.show()

# momentum grid
# momem´ntum step size dp = 2.0*pi/(nx*dx), pmax=pi/dx
# pmin = - pmax
pmin = -np.pi/dx
pmax = np.pi/dx
dp = 2.*np.pi/(nx*dx)
#dp = 1.0
#for i in range(0, nx):
#        kx[i] = pmin+ i*dp
#        T[i] = kx[i]*kx[i]/2.0
# That is the routine to compute 
# the momemtum and kinetic energy arrays for FFT
# 
# stepsize for the kicking strength dP
# Pmin = 0.5, Pmax = 6, np = int((Pmax-Pmin)/dP)
# dP = 0.1 ---> np = (6-0.5)/0.1= 55
dP = 0.1
Pmin = 0.5
Pmax = 6.0
npp = int((Pmax-Pmin)/dP)
print(npp)

for i in range(0, nx):
    if i < nx/2:
        kx[i] = i*dp
        T[i] = kx[i]*kx[i]/2.0
    else:
        kx[i] = (i-nx)*dp
        T[i] = kx[i]*kx[i]/2.0

AccelerationRateP = []; StrengthP = []
for i in np.arange(0,npp+1):
    P = Pmin + i*dP
    # Initial state as homogeneous wavefunction psi(x,0)=1/sqrt(2.*pi)
    # 1/sqrt(2.np.pi) is in the form of zero-momemtum plane wave 
    # (p.36, Phys. Rep.77, 538 (2014))
    normPsi = 0.0
    for i in range(0, nx):
        psi[i] = 1./np.sqrt(2.*np.pi)
    #    psi[i] = np.exp(-0.25*x[i]*x[i]) # inhomogeneous wavefunction
        normPsi = normPsi + np.abs(psi[i])**2.*dx
    #print(normPsi)
    # Normalized initial wavefunction
    normPsi0 = 0.0
    for i in range(0, nx):
        psi[i] = psi[i]/np.sqrt(normPsi)
        normPsi0 = normPsi0 + np.abs(psi[i])**2.*dx
        densPsi[i] = np.abs(psi[i])**2.
    print(normPsi0)
    # Here our homogenoeus wavefunction was already normalized
    
    #plt.xlabel('position x')
    #plt.ylabel('|Psi(x)|^2')
    #plt.plot(x,densPsi)
    #plt.show()
    
    # propagation using the unitary operator
    # U = exp(-iu*dt*V/(2.*hbar)*exp(-iu*dt*T/hbar)*exp(-iu*dt*V/(2.*hbar))
    Current =[]; Energy =[] ; Time =[]
    for m in range(1,mmax+1):
        psi1 = np.exp(-iu*dt*P*Vratchet)*psi
        psibar1 = np.fft.fft(psi1)
        psibar2 = np.exp(-iu*dt*T*kbar)*psibar1
        psi = np.fft.ifft(psibar2)
        
        normPsi = 0.0
        for i in range(0, nx):
            normPsi = normPsi + np.abs(psi[i])**2.*dx
    #    print(normPsi)
    
        normPsi0 = 0.0
        for i in range(0,nx):
            psi[i] = psi[i]/np.sqrt(normPsi)
            normPsi0 = normPsi0 + np.abs(psi[i])**2.*dx
            densPsi[i] = np.abs(psi[i])**2.
    #    print(normPsi0)
        
    #    plt.xlabel('position x')
    #    plt.ylabel('|Psi(x)|^2')
    #    plt.plot(x,densPsi)
    #    plt.show()
        
        sumk1 = complex(0,0); sumk2 = complex(0,0)
        #psi[nx] = complex(0,0)
        for j in range(1, nx-1):
            d1Psi[j] = (psi[j+1]-psi[j])/dx
            d2Psi[j] = (psi[j+1]-2.*psi[j]+psi[j-1])/(dx*dx)
            sumk1 += d1Psi[j]*np.conj(psi[j])*dx*(-iu)
            sumk2 += d2Psi[j]*np.conj(psi[j])*dx*(-1.0)
        
    #    Current.append(abs(sumk1))
    #    Energy.append(abs(sumk2))
#        Current.append(np.real(sumk1))
#        Energy.append(np.real(sumk2))
#        Time.append(m)
#        
#    plt.xlabel('time t')
#    plt.ylabel('Energy <k^2>')
#    plt.plot(Time,Energy)
#    plt.legend('ooo')
#    plt.show()
#    
#    plt.xlabel('time t')
#    plt.ylabel('current <k>')
#    plt.plot(Time,Current)
#    plt.legend('---')
#    plt.show()
    AccelerationRateP.append(np.real(sumk1)/mmax)
    StrengthP.append(P)

plt.xlabel('strength P')
plt.ylabel('Acceleration Rate <k>/nkick')
plt.plot(StrengthP,AccelerationRateP)
plt.legend('---')
plt.show()



