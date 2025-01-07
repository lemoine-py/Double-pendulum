""" 
This code aims to compute the largest Lyapunov exponents of the double pendulum system

1. Defining the Jacobian matrix
2. Runge-Kutta 4 for matrices

3. Local Lyapunov exponents
4. Global Lyapunov exponents
5. Graphs

"""

# Libraries
import numpy as np
import scipy as sp # For the matrix exponential
import matplotlib.pyplot as plt
from tqdm import tqdm # Progression bar

# math labels
lambda_math = r"$\lambda_{max}$"
lambda_n_math = r"$\lambda_{max,n}$"
delta_math = r"$\delta x_0$"
t_max_math = r"$t_{max}$"
t_step_math = r"$t_{step}$"

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

# Time parameters
t_step = 20
dt = 0.01
N = int(np.floor(t_step/dt))+1 # about 2000 timesteps

theta_N = 30
n_step = 50

# Initial dx0
dx0 = np.array([0, 0, 1, 0])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # so that norm(dx0) = 10**(-10)

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

#-------------------------------------------

th = np.linspace(0,np.pi,30) # Angle array

# Templates for initial conditions (u_0)
theta_theta = [np.array([0, 0, th[p], th[p]])  for p in range(30)]
theta_minustheta = [np.array([0, 0, th[p], -th[p]]) for p in range(30)]
theta_zero = [np.array([0, 0, th[p], 0]) for p in range(30)]
zero_theta = [np.array([0, 0, 0, th[p]]) for p in range(30)]

def global_lyaps(u_0, dx0, dt, last_RK4, F_deriv, lyapunov, n_step, theta_N):
    """Computes the global largest lyapunov exponents for different initial conditions"""

    lya_th = np.zeros(30) # "Lyapunov for each angle" array

    steps = 30*n_step
    with tqdm(total=steps) as pbar: # Progression bar
        for p in range(30):
            u0 = u_0[p]
            lya_steps = np.zeros(n_step)
            for i in range(n_step):
                u1 = last_RK4(F_deriv, u0, dt, N) # updating th1, th2, w1, w2
                lya_steps[i], dx = lyapunov(u0, dx0, t_step) # computing the lyapunov exponent
                u0 = u1 # updating initial conditions
                dx0 = dx/(np.linalg.norm(dx)*10**10) # Renormalising so that norm(dx0) = 10**(-10)
                pbar.update(1)  # Updates the progression bar
            lya_th[p] = np.average(lya_steps) 
            # filtered_lya_th = np.array(lya_th)[np.isfinite(lya_th)] # Filtering the NaN and inf values
            # lya_th_ave[p] = np.average(filtered_lya_th)
    return lya_th

def graph_lyaps(th, lya_th, u_0):
    """ Plots the Lyapunov exponents for different initial conditions"""

    if all(np.array_equal(a, b) for a, b in zip(u_0, theta_theta)):
        title = r"Initial conditions: $\theta_{1,0}= \theta_{2,0} = \theta$ ; $\omega_{1,0} = \omega_{2,0} = 0$"
        filename = "global_lyap_theta_theta.png"
    elif all(np.array_equal(a, b) for a, b in zip(u_0, theta_minustheta)):
        title = r"Initial conditions: $\theta_{1,0} = \theta$, $\theta_{2,0} = -\theta$ ; $\omega_{1,0} = \omega_{2,0} = 0$"
        filename = "global_lyap_theta_minustheta.png"
    elif all(np.array_equal(a, b) for a, b in zip(u_0, theta_zero)):
        title = r"Initial conditions: $\theta_{1,0} = \theta$, $\theta_{2,0} = 0$ ; $\omega_{1,0} = \omega_{2,0} = 0$"
        filename = "global_lyap_theta_zero.png"
    elif all(np.array_equal(a, b) for a, b in zip(u_0, zero_theta)):
        title = r"Initial conditions: $\theta_{1,0} = 0$, $\theta_{2,0} = \theta$ ; $\omega_{1,0} = \omega_{2,0} = 0$"
        filename = "global_lyap_zero_theta.png"
    
    plt.figure()
    plt.plot(th, lya_th, label = f"{lambda_n_math} average")
    plt.suptitle(title)
    plt.ylabel("Lyapunov exponent")
    plt.xlabel(r"$\theta$ (rad)")
    plt.text(0.2, 0.7, f"dt = {dt} \n {t_max_math} = n*{t_step_math} = 50*20\n {delta_math} = [0,0,1e-10,0]", bbox = dict(facecolor = "white", alpha = 1), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.grid()
    plt.savefig(filename)

lyap_theta_theta = global_lyaps(theta_theta, dx0, dt, last_RK4, F_deriv, lyapunov, n_step, theta_N)
#lyap_theta_minustheta = global_lyaps(theta_minustheta, dx0, dt, last_RK4, F_deriv, lyapunov, n_step, theta_N)
#lyap_theta_zero = global_lyaps(theta_zero, dx0, dt, last_RK4, F_deriv, lyapunov, n_step, theta_N)
#lyap_zero_theta = global_lyaps(zero_theta, dx0, dt, last_RK4, F_deriv, lyapunov, n_step, theta_N)

graph_lyaps(th, lyap_theta_theta, theta_theta)
#graph_lyaps(th, lyap_theta_minustheta, theta_minustheta)
#graph_lyaps(th, lyap_theta_zero, theta_zero)
#graph_lyaps(th, lyap_zero_theta, zero_theta)

plt.show()