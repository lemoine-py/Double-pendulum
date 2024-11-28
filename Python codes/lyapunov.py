# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:59:44 2024

@author: Solal
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81



### Jacobian ### -------------------------------------------------------------------------------------
def jacobian(w1, w2,th1, th2):
    jac = np.zeros((4,4))
    
    jac[0][2] = 1 #dth1/dw1
    jac[1][3] = 1 #dth2/w2
    jac[2][2] = -2*np.sin(th1-th2)*m2*2*w1*l1*np.cos(th1-th2)/(l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[2][3] = -2*np.sin(th1-th2)*m2*2*w2*l2/(l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[3][2] = 2*np.sin(th1-th2)*2*w1*l1*(m1+m2)/(l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[3][3] = 2*np.sin(th1-th2)*2*w2*l2*m2*np.cos(th1-th2)/(l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    
    num20 = (-g*(2*m1+m2)*np.cos(th1)-m2*g*np.cos(th1-2*th2)-2*np.cos(th1-th2)*m2*w2**2*l2-2*m2*np.cos(th1-th2)**2*w1**2*l1+2*np.sin(th1-th2)**2*m2*w1**2*l1)*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2)))*(l1*m2*2*np.sin(2*th1-2*th2))
    denom20 = (l1*(2*m2+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[2][0] = num20/denom20
    
    num21 = (2*m2*g*np.cos(th1-2*th2)+2*np.cos(th1-th2)*m2*w2**2*l2+2*np.cos(th1-th2)**2*m2*l2*w1**2-2*np.sin(th1-th2)**2*m2*w1**2*l1)*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2*w1**2*l2*np.cos(th1-th2)))*(-l1*m2*2*np.sin(2*th1-2*th2))
    denom21 = (l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[2][1] = num21/denom21
    
    num30 = 2*np.cos(th1-th2)*w1**2*l1*(m1+m2)+2*g*(m1+m2)*(np.cos(th1-th2)*np.cos(th1)-np.sin(th1-th2)*np.sin(th1)+2*w2**2*l2*m2*(np.cos(th1-th2)**2-np.sin(th1-th2)**2))*l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2)))*l1*(2*m2*np.sin(2*th1-2*th2))
    denom30 = (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[3][0] = num30/denom30
    
    num31 = -2*np.cos(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1))+2*w2**2*l2*m2*m2*(np.sin(th1-th2)**2-np.cos(th1-th2**2))*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2)))*(-2*l2*m2*np.sin(2*th1-2*th2))
    denom31 = (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[3][1] = num31/denom31
                                                                                   
    return jac

### Runge-kutta 4 for matrices

def rk4_matrix(dt,N,A):
    y = np.zeros((4, N+1))
    y[:, 0] = np.array([0, 0, 10**(-10), 0]) # delta's initial conditions
    for i in range(N):
        k1 = dt * (A @ y[:, i])
        k2 = dt * (A @ (y[:, i] + k1/2))
        k3 = dt * (A @ (y[:, i] + k2/2))
        k4 = dt * (A @ (y[:, i] + k3))
        y[:, i+1] = y[:, i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

### Spin-up phase ### -------------------------------------------------------------------------------------

# initial conditions
w1_0 = 0
w2_0 = 0
th1_0 = 0.1
th2_0 = np.pi

n = 10000
t_max = 10000
dt = 1
t_delta = np.linspace(0, t_max, n)

delta_0 = [0, 0, 10**(-10), 0]

J = jacobian(w1_0, w2_0, th1_0, th2_0)

#-------------------------------------------
delta = rk4_matrix(dt, n, J)

def lya_exp(t, delta):
    lya = np.zeros(n)
    norm = np.zeros(n)
    for i in range (n):
        norm[i] = np.linalg.norm(delta[:,i])
    
    for i in range(n):
        lya[i] = np.log((norm[i])/norm[0])/t[i]
    return lya

lya = lya_exp(t_delta, delta)
#-------------------------------------------
delta_a = np.zeros((n,4))
for i in range(n):
    delta_a[i] = sp.linalg.expm(J*t_delta[i]) @ delta_0

def lya_a_exp(t, delta):
    lya = np.zeros(n)
    norm = np.zeros(n)
    for i in range (n):
        norm[i] = np.linalg.norm(delta[i])
    
    for i in range(n):
        lya[i] = np.log((norm[i])/norm[0])/t[i]
    return lya

lya_a = lya_a_exp(t_delta, delta_a)
#-------------------------------------------

plt.figure()
plt.plot(t_delta, lya)
plt.plot(t_delta, lya_a)
plt.suptitle("Lyapunov exponent")
plt.xlabel("Time (s)")
plt.ylabel("Lyapunov exponent")
plt.grid()
plt.show()
