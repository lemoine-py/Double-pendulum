"""
This script calculates the initial energy for each pair of initial angles
(th1, th2) in the range [0, pi] x [0, pi] and plots the results as a colormap.

omega1 and omega2 are set to 0.
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Progression bar

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

w1_0 = 0
w2_0 = 0

def initial_energy(m1, m2, l1, l2, g, w1, w2, th1, th2):
    """ Computes the total energy of the double pendulum for given parameters """
    T = 0.5*m1*l1**2*w1**2 + 0.5*m2*(l1**2*w1**2 + l2**2*w2**2 + 2*l1*l2*w1*w2*np.cos(th1-th2)) # kinetic energy
    V = -(m1 + m2)*g*l1*np.cos(th1) - m2*g*l2*np.cos(th2) # potential energy
    E =  T + V # total energy
    return E

theta_N = 314 # Discritization of the angle
th = np.linspace(0,np.pi,theta_N) # Angle array
energy_pq_th = np.zeros((theta_N,theta_N))

with tqdm(total=theta_N*theta_N) as pbar: # Progression bar
    for p in range(theta_N):
        for q in range(theta_N):
            energy_pq_th[p][q] = initial_energy(m1, m2, l1, l2, g, w1_0, w2_0, th[p], th[q])
            pbar.update(1)

# Plot the array as a colormap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(energy_pq_th, cmap='jet', extent=[0, np.pi, 0, np.pi], origin='lower')

ax.set_xlabel(r'$\theta_2$')
ax.set_ylabel(r'$\theta_1$')

cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Energy')

plt.title(r'Initial energy for each ($\theta_1$,$\theta_2$)')

plt.savefig("energy_cmap.png")

plt.show()