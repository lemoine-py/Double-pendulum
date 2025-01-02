""" 
This code is the main complete code for the Double Pendulum project.
By "complete", we mean that this code is self-sufficient and can be run as is.

This code simulates the dynamics of one double pendulum system 
and produces 5 files:

1. <angles_velocities.png> : four subplots for each angles and angular velocities with respect to time
2. <one_brownian_motion.png> : theta_1 vs theta_2 plot, illustrates the brownian motion of the system
3. <one_energy_loss.png> : total energy loss due to the numerical model's inaccuracy
4. <one_XY_paths.png> : 2D-cartesian trajectories of the pendulums
5. <one_pendulum.gif> : animation of the four pendula's movements (20 seconds)

"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81

# Time parameters and discretization
t_max = 20
h = 0.02 # Not smaller than that for the gif
N = int(np.floor(t_max/h))+1
t = np.linspace(0, t_max, N)

# Initial conditions
w1_0 = 0
w2_0 = 0
th1_0 = np.pi
th2_0 = np.pi
u0 = np.array([w1_0, w2_0, th1_0, th2_0])

"""
from Initialize_planet import Initialize, planet

# Lets the user choose the parameters and initial conditions by a series of prompts
g, l1, l2, m1, m2, th1_0, th2_0, w1_0, w2_0 = Initialize(planet)
"""

### Defining functions to solve the differential equation
def F_deriv(w1, w2, th1, th2): 
    """ Derivatives of the u array """

    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2    
    
    return np.array([w1_dot, w2_dot, w1, w2])

def solve_RK4(f, u0, h):
    """ Solves the differential equation using the Runge-Kutta 4th order method """

    u = np.zeros((N,4))
    u[0] = u0
    for i in range(N - 1):
        k1 = h * f(u[i][0], u[i][1], u[i][2], u[i][3])
        k2 = h * f(u[i][0] + k1[0]/2, u[i][1] + k1[1]/2, u[i][2] + k1[2]/2, u[i][3] + k1[3]/2)
        k3 = h * f(u[i][0] + k2[0]/2, u[i][1] + k2[1]/2, u[i][2] + k2[2]/2, u[i][3] + k2[3]/2)
        k4 = h * f(u[i][0] + k3[0], u[i][1] + k3[1], u[i][2] + k3[2], u[i][3] + k3[3])
        u[i+1] = u[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, u

### Solving the differential equation and storing the solutions in u1
t, u = solve_RK4(F_deriv, u0, h)

th1 = u[:,2] 
th2 = u[:,3]
w1 = u[:,0]
w2 = u[:,1]

# Plotting the solutions i.e. all 4 components of u
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(t, w1)
ax[0, 0].set_xlabel("Time (s)")
ax[0, 0].set_ylabel("Angular velocity (rad/s)")
ax[0, 0].set_title(r"$\omega_1$")
ax[0, 0].grid()
    
ax[0, 1].plot(t, w2)
ax[0, 1].set_xlabel("Time (s)")
ax[0, 1].set_ylabel("Angle (rad)")
ax[0, 1].set_title(r"$\omega_2$")
ax[0, 1].grid()
    
ax[1, 0].plot(t, th1)
ax[1, 0].set_xlabel("Time (s)")
ax[1, 0].set_ylabel("Angular velocity (rad/s)")
ax[1, 0].set_title(r"$\theta_1$")
ax[1, 0].grid()
    
ax[1, 1].plot(t, th2)
ax[1, 1].set_xlabel("Time (s)")
ax[1, 1].set_ylabel("Angle (rad)")
ax[1, 1].set_title(r"$\theta_2$")
ax[1, 1].grid()

fig.tight_layout()
fig.savefig("angles_velocities.png")


# Brownian motion plot (theta1 vs theta2)
fig, ax = plt.subplots()
ax.plot(th1, th2)
ax.plot(th1[0], th2[0], "-o", label = "Starting point", color = "red")
ax.set_title(r"Parametric plot - $\theta_1$ vs $\theta_2$")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.legend()
ax.grid()
fig.savefig("one_brownian_motion.png")


### Energy calculations
def energy(m1, m2, l1, l2, g, w1, w2, th1, th2):
    """ Computes the total energy of the double pendulum for given parameters """
    T = np.zeros(N) # kinetic energy
    V = np.zeros(N) # potential energy
    E = np.zeros(N) # total energy
    for i in range(N):
        T[i] = 0.5*m1*l1**2*w1[i]**2 + 0.5*m2*(l1**2*w1[i]**2 + l2**2*w2[i]**2 + 2*l1*l2*w1[i]*w2[i]*np.cos(th1[i]-th2[i])) 
        V[i] = -(m1 + m2)*g*l1*np.cos(th1[i]) - m2*g*l2*np.cos(th2[i]) 
        E[i] = T[i] + V[i] 
    return T, V, E

T, V, E = energy(m1, m2, l1, l2, g, w1, w2, th1, th2)

# Plotting the difference between the initial total energy and the computed total energy along time
plt.figure()
plt.plot(t, E[0]-E)
plt.suptitle("Total energy loss due to RK4 inacurracy")
plt.xlabel("Time [s]")
plt.ylabel("E(t=0) - E(t) [J]")
plt.grid()
plt.savefig("one_energy_loss.png")


### Position in cartesian coordinates
def cartesian(tha, thb, la, lb):
    """ Computes the cartesian coordinates in 2D for each mass """
    xa = np.zeros(N) # x component of m1
    ya = np.zeros(N) # y component of m1

    xb = np.zeros(N) # x component of m2
    yb = np.zeros(N) # y component of m2

    for i in range(N):
        xa[i] = la*np.sin(tha[i])
        ya[i] = -la*np.cos(tha[i])
        
        xb[i] = xa[i] + lb*np.sin(thb[i])
        yb[i] = ya[i] - lb*np.cos(thb[i])
    return xa, ya, xb, yb

x1, y1, x2, y2 = cartesian(th1, th2, l1, l2)

# Plot the paths of the pendula
fig, ax = plt.subplots()
ax.plot(x1, y1, label="m1")
ax.plot(x2, y2, label="m2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Double pendulum trajectories")
ax.legend()
ax.set_aspect('equal')
fig.savefig("one_XY_paths.png")

plt.show() # Shows at once every plot that has been produced before (in multiple windows)

### ANIMATION ###
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(3*(l1+l2),3*(l1+l2)))
ax.set_facecolor('k')
#ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
#ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = ax.plot([], [], 'o-', lw=3, markersize=8, color = "silver")
ax.plot(0, 0, 'o', color = "white")  # origin
ax.set_ylim(-1.5*(l1+l2),1.5*(l1+l2))
ax.set_xlim(-1.5*(l1+l2),1.5*(l1+l2))
ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50)
ani.save('one_pendulum.gif',writer='pillow',fps=1/h) # Save the animation as a gif