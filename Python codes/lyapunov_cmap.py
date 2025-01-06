"""
This script calculates the global Lyapunov exponents for each pair of initial angles
(θ1, θ2) in the range [0, π] x [0, π] and plots the results as a colormap.

"""

# Libraries
import numpy as np
import scipy as sp # For the matrix exponential
import matplotlib.pyplot as plt
from tqdm import tqdm # Progression bar

# Latex labels
lambda_latex = r"$\lambda_{max}$"
lambda_n_latex = r"$\lambda_{max,n}$"
delta_latex = r"$\delta x_0$"
t_max_latex = r"$t_{max}$"
t_step_latex = r"$t_{step}$"

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

# Time parameters
t_step = 20
th_step = 30
global_step = 25
dt = 0.01
N = int(np.floor(t_step/dt))+1 # about 2000 timesteps

# Initial dx0
dx0 = np.array([0, 0, 1, 1])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # so that norm(dx0) = 10**(-10)

def F_deriv(w1, w2, th1, th2):
    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2    
    
    return np.array([w1_dot, w2_dot, w1, w2])
    
def solve_RK4(f, u0, dt, N):
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



th = np.linspace(0,np.pi,th_step) # Angle array
lya_pq_th = np.zeros((th_step,th_step))

with tqdm(total=th_step*th_step*global_step) as pbar: # Progression bar
    for p in range(th_step):
        for q in range(th_step):
            Jpq = jacobian(0, 0, th[p], th[q])
            u0 = np.array([0, 0, th[p], th[q]])
            lya_pq = np.zeros(global_step)
            for i in range(global_step):
                u1 = solve_RK4(F_deriv, u0, dt, N)
                lya_pq[i], dx = lyapunov(u0, dx0, t_step)
                u0 = u1
                dx0 = dx/(np.linalg.norm(dx)*10**10)
                pbar.update(1)

            lya_pq_th[p][q] = np.average(lya_pq)

# Plot the array as a colormap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(lya_pq_th, cmap='jet', extent=[0, np.pi, 0, np.pi], origin='lower')

# Set axis labels and ticks
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')

# Add colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Lyapunov exponent')

# Set axis ticks
#ax.set_xticks(np.linspace(0, np.pi, 6))
#ax.set_yticks(np.linspace(0, np.pi, 6))
#ax.set_xticklabels([f'{tick:.2f}' for tick in np.linspace(0, np.pi, 6)])
#ax.set_yticklabels([f'{tick:.2f}' for tick in np.linspace(0, np.pi, 6)])

# Show the plot
plt.title(r'$\lambda_{max}$ for each ($\theta_1$,$\theta_2$), starting from $\delta x_0$ = [0,0,1,1]e-10')

plt.savefig("global_lyap_cmap_z.png")

print()
print(f"t_step = {t_step}")
print(f"th_step = {th_step}")
print(f"global_step = {global_step}")
print(f"dt = {dt}")
print()

plt.show()