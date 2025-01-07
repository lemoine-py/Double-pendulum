""" 
This script is a collection of various functions used in this project. 
It could be used as a library for other scripts, but the structure of the code should be then modified.

The functions are:
- F_deriv : computes the derivatives of the double pendulum's system of differential equations
- last_RK4 : computes the last set of values of a system of differential equations using the Runge-Kutta 4 method
- RK4_matrix : computes the values of a matrix differential equation using the Runge-Kutta 4 method
- jacobian : computes the Jacobian of the double pendulum's system of differential equations
- lyapunov : computes the largest Lyapunov exponent of the double pendulum's system of differential equations
- lyapunov_RK4 : computes the Lyapunov exponents by first solving for delta_x using the Runge-Kutta 4 method
- lyapunov_expm : computes the Lyapunov exponents by first solving for delta_x using the matrix exponential method
"""

# Libraries
import numpy as np
import scipy as sp # For the matrix exponential
import matplotlib.pyplot as plt
from tqdm import tqdm # Progression bar

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

def F_deriv(w1, w2, th1, th2):
    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2    
    
    return np.array([w1_dot, w2_dot, w1, w2])

def last_RK4(f, u0, dt, N):
    """ Returns : u (4x1 array of last computed values)"""
    u = np.zeros(4)
    u = u0
    for i in range(N - 1):
        k1 = dt * f(u[0], u[1], u[2], u[3])
        k2 = dt * f(u[0] + k1[0]/2, u[1] + k1[1]/2, u[2] + k1[2]/2, u[3] + k1[3]/2)
        k3 = dt * f(u[0] + k2[0]/2, u[1] + k2[1]/2, u[2] + k2[2]/2, u[3] + k2[3]/2)
        k4 = dt * f(u[0] + k3[0], u[1] + k3[1], u[2] + k3[2], u[3] + k3[3])
        u = u + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u # last computed set of 4 values

def RK4_matrix(A, delta_0, dt, N):
    """ Runge-Kutta 4 scheme for F = A@y 
        Returns y (ndarray(4,N+1))
    """
    y = np.zeros((4, N+1))
    y[:, 0] = delta_0 # delta's initial conditions
    for i in range(N):
        k1 = dt * (A @ y[:, i])
        k2 = dt * (A @ (y[:, i] + k1/2))
        k3 = dt * (A @ (y[:, i] + k2/2))
        k4 = dt * (A @ (y[:, i] + k3))
        y[:, i+1] = y[:, i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

def jacobian(w1, w2, th1, th2):
    jac = np.zeros((4,4))
   
    jac[0][2] = 1 #dth1/dw1
    jac[1][3] = 1 #dth2/w2
    jac[2][2] = -4*m2*w1*np.sin(th1 - th2)*np.cos(th1 - th2)/(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)
    jac[2][3] = -4*l2*m2*w2*np.sin(th1 - th2)/(l1*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
    jac[3][2] = 4*l1*w1*(m1 + m2)*np.sin(th1 - th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
    jac[3][3] = 4*m2*w2*np.sin(th1 - th2)*np.cos(th1 - th2)/(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)
    jac[2][0] = -2*m2*(-g*m2*np.sin(th1 - 2*th2) - g*(2*m1 + m2)*np.sin(th1) - 2*m2*(l1*w1**2*np.cos(th1 - th2) + l2*w2**2)*np.sin(th1 - th2))*np.sin(2*th1 - 2*th2)/(l1*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)**2) + (-g*m2*np.cos(th1 - 2*th2) - g*(2*m1 + m2)*np.cos(th1) + 2*l1*m2*w1**2*np.sin(th1 - th2)**2 - 2*m2*(l1*w1**2*np.cos(th1 - th2) + l2*w2**2)*np.cos(th1 - th2))/(l1*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
    jac[2][1] = 2*m2*(-g*m2*np.sin(th1 - 2*th2) - g*(2*m1 + m2)*np.sin(th1) - 2*m2*(l1*w1**2*np.cos(th1 - th2) + l2*w2**2)*np.sin(th1 - th2))*np.sin(2*th1 - 2*th2)/(l1*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)**2) + (2*g*m2*np.cos(th1 - 2*th2) - 2*l1*m2*w1**2*np.sin(th1 - th2)**2 + 2*m2*(l1*w1**2*np.cos(th1 - th2) + l2*w2**2)*np.cos(th1 - th2))/(l1*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
    jac[3][0] = -4*m2*(g*(m1 + m2)*np.cos(th1) + l1*w1**2*(m1 + m2) + l2*m2*w2**2*np.cos(th1 - th2))*np.sin(th1 - th2)*np.sin(2*th1 - 2*th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)**2) + 2*(-g*(m1 + m2)*np.sin(th1) - l2*m2*w2**2*np.sin(th1 - th2))*np.sin(th1 - th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)) + 2*(g*(m1 + m2)*np.cos(th1) + l1*w1**2*(m1 + m2) + l2*m2*w2**2*np.cos(th1 - th2))*np.cos(th1 - th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
    jac[3][1] = 2*m2*w2**2*np.sin(th1 - th2)**2/(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2) + 4*m2*(g*(m1 + m2)*np.cos(th1) + l1*w1**2*(m1 + m2) + l2*m2*w2**2*np.cos(th1 - th2))*np.sin(th1 - th2)*np.sin(2*th1 - 2*th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2)**2) - 2*(g*(m1 + m2)*np.cos(th1) + l1*w1**2*(m1 + m2) + l2*m2*w2**2*np.cos(th1 - th2))*np.cos(th1 - th2)/(l2*(2*m1 - m2*np.cos(2*th1 - 2*th2) + m2))
                                                                                   
    return jac

def lyapunov(u, dx, t_step):
    """ Computing lambda_max after t_step seconds, starting from a given u and dx"""
    J = jacobian(u[0],u[1],u[2],u[3]) # evaluating the jacobian at u
    dxf = sp.linalg.expm(J*t_step) @ dx # solving for dx after t seconds
    norm = np.linalg.norm(dxf)
    lyapf = np.log(norm/np.linalg.norm(dx))*1/t_step # Lyapunov exponent after t seconds
    return lyapf, dxf

def lyapunov_RK4(t, delta, N):
    lya = np.zeros(N)
    norm = np.zeros(N)
    for i in range (N):
        norm[i] = np.linalg.norm(delta[:,i])
    for i in range(N):
        lya[i] = np.log((norm[i])/norm[0])/t[i]
    return lya

def lyapunov_expm(t_delta, delta_a, N):
    lya = np.zeros(N)
    norm = np.zeros(N)
    for i in range (N):
        norm[i] = np.linalg.norm(delta_a[i])
    
    for i in range(N):
        lya[i] = np.log((norm[i])/norm[0])/t_delta[i]
    return lya