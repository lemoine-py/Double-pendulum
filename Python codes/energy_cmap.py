"""
This script calculates the initial energy for each pair of initial angles
(θ1, θ2) in the range [0, π] x [0, π] and plots the results as a colormap.

omega1 and omega2 are set to 0.

"""

# Libraries
import numpy as np
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

w1_0 = 0
w2_0 = 0

def energy(m1, m2, l1, l2, g, w1, w2, th1, th2):
    """ Computes the total energy of the double pendulum for given parameters """
    T = 0.5*m1*l1**2*w1**2 + 0.5*m2*(l1**2*w1**2 + l2**2*w2**2 + 2*l1*l2*w1*w2*np.cos(th1-th2)) # kinetic energy
    V = -(m1 + m2)*g*l1*np.cos(th1) - m2*g*l2*np.cos(th2) # potential energy
    E =  T + V # total energy
    return E

th_step = 314 # Discritization of the angle
th = np.linspace(0,np.pi,th_step) # Angle array
energy_pq_th = np.zeros((th_step,th_step))

with tqdm(total=th_step*th_step) as pbar: # Progression bar
    for p in range(th_step):
        for q in range(th_step):
            energy_pq_th[p][q] = energy(m1, m2, l1, l2, g, w1_0, w2_0, th[p], th[q])
            pbar.update(1)

# Plot the array as a colormap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(energy_pq_th, cmap='jet', extent=[0, np.pi, 0, np.pi], origin='lower')

ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')

cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Energy')

plt.title(r'Initial energy for each ($\theta_1$,$\theta_2$)')

plt.savefig("energy_cmap.png")

plt.show()