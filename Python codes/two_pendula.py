""" 
This code aims to compare four pendula with different initial conditions.
It simulates the dynamics of two double pendulum systems and plots the results.

This code produces 6 files : 

1. <angles1_velocities1.png> : four subplots for each angles and angular velocities with respect to time
2. <angles2_velocities2.png> : same as 1. but for another double pendulum
3. <four_brownian_motion.png> : theta_1 vs theta_2 plot, illustrates the brownian motion of the system
4. <four_energy_loss.png> : total energy losses due to the numerical model's inaccuracy
5. <four_XY_paths.png> : four subplots of 2D-cartesian trajectories for each double pendulum
6. <four_pendula.gif> : animation of the four pendulums' movements (20 seconds)

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

### Initial conditions

# Pendulum 1
w1_0 = 0
w2_0 = 0
th1_0 = np.pi
th2_0 = 1.4
u0 = np.array([w1_0, w2_0, th1_0, th2_0])
# Pendulum 2
w11_0 = 0
w22_0 = 0
th11_0 = np.pi
th22_0 = 1.5
u00 = np.array([w11_0, w22_0, th11_0, th22_0])

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

def solve_RK4(f, u0, h, t):
    """ Solves the differential equation using the Runge-Kutta 4th order method """
    u = np.zeros((N,4))
    u[0] = u0
    for i in range(N - 1):
        k1 = h * f(u[i][0], u[i][1], u[i][2], u[i][3])
        k2 = h * f(u[i][0] + k1[0]/2, u[i][1] + k1[1]/2, u[i][2] + k1[2]/2, u[i][3] + k1[3]/2)
        k3 = h * f(u[i][0] + k2[0]/2, u[i][1] + k2[1]/2, u[i][2] + k2[2]/2, u[i][3] + k2[3]/2)
        k4 = h * f(u[i][0] + k3[0], u[i][1] + k3[1], u[i][2] + k3[2], u[i][3] + k3[3])
        u[i+1] = u[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u

### Solving the differential equation and storing the solutions in u1
u1 = solve_RK4(F_deriv, u0, h, t)
u11 = solve_RK4(F_deriv, u00, h, t)

# Pendulum 1
th1 = u1[:,2] 
th2 = u1[:,3]
w1 = u1[:,0]
w2 = u1[:,1]
# Pendulum 2
th11 = u11[:,2]
th22 = u11[:,3]
w11 = u11[:,0]
w22 = u11[:,1]

### Energy calculations ### --------------------------------------------------------------------------------

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

T1, V1, E1 = energy(m1, m2, l1, l2, g, w1, w2, th1, th2)
T2, V2, E2 = energy(m1, m2, l1, l2, g, w11, w22, th11, th22)

# Plotting the difference between the initial total energy and the computed total energy along time
plt.figure()
plt.plot(t, E1[0]-E1, label = "Pendulum 1")
plt.plot(t, E2[0]-E2, label = "Pendulum 2")
plt.suptitle("Total energy losses due to RK4 inacurracy")
plt.xlabel("Time [s]")
plt.ylabel("E(t=0) - E(t) [J]")
plt.legend()
plt.grid()
#plt.savefig("two_energy_loss.png")

### ---------------------------------------------------------------------------------------------------------
# Plotting the solutions i.e. all 4 components of u
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(t, w1, color = "magenta")
ax[0, 0].set_xlabel("Time (s)")
ax[0, 0].set_title(r"$\omega_1$")
ax[0, 0].grid()
    
ax[0, 1].plot(t, w2, color = "magenta")
ax[0, 1].set_xlabel("Time (s)")
ax[0, 1].set_title(r"$\omega_2$")
ax[0, 1].grid()
    
ax[1, 0].plot(t, th1, color = "magenta")
ax[1, 0].set_xlabel("Time (s)")
ax[1, 0].set_title(r"$\theta_1$")
ax[1, 0].grid()
    
ax[1, 1].plot(t, th2, color = "magenta")
ax[1, 1].set_xlabel("Time (s)")
ax[1, 1].set_title(r"$\theta_2$")
ax[1, 1].grid()

fig.tight_layout()
fig.savefig("angles1_velocities1.png")

### Brownian Motion plot (th1 vs th2) ###
fig, ax = plt.subplots()
ax.plot(th1, th2, label = "Pendulum 1")
ax.plot(th11, th22, label = "Pendulum 2")
ax.plot(th1[0], th2[0], "-o", label = "Starting point", color = "red")
ax.set_title("Parametric plot of the pendulum's angles")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_aspect('equal')
ax.legend()
ax.grid()
fig.savefig("two_brownian_motion.png")


### Positions in cartesian coordinates ---------------------------------------------------------------------

def cartesian(tha, thb, la, lb):
    """ Computes the cartesian coordinates for both masses """
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
x11, y11, x22, y22 = cartesian(th11, th22, l1, l2)

# Plot the paths of the pendula
figa, axa = plt.subplots(2)
axa[0].plot(x1, y1, label="m1")
axa[0].plot(x2, y2, label="m2", color = "magenta")
axa[0].set_xlabel("x")
axa[0].set_ylabel("y")
axa[0].set_title("Pendulum 1")
axa[0].legend()
axa[0].set_aspect('equal')

axa[1].plot(x11, y11, label="m1")
axa[1].plot(x22, y22, label="m2", color = "cyan")
axa[1].set_xlabel("x")
axa[1].set_ylabel("y")
axa[1].set_title("Pendulum 2")
axa[1].legend()
axa[1].set_aspect('equal')

figa.tight_layout()
figa.savefig("two_XY_paths.png")

plt.show() # Shows at once every plot that has been produced before (in multiple windows)


### ANIMATION GIF ###

fig, ax = plt.subplots(figsize=(3 * (l1 + l2), 3 * (l1 + l2)))
ax.set_facecolor('k')
ax.set_xlim(-1.5 * (l1 + l2), 1.5 * (l1 + l2))
ax.set_ylim(-1.5 * (l1 + l2), 1.5 * (l1 + l2))
#ax.plot(0, 0, 'o', color = "green")  # Origin

# Initialization of the pendulums
ln1, = ax.plot([], [], 'o-', lw=3, markersize=8, color="magenta", label="Pendulum 1")
ln2, = ax.plot([], [], 'o-', lw=3, markersize=8, color="cyan", label="Pendulum 2")
ax.legend()

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])  # Pendulum 1
    ln2.set_data([0, x11[i], x22[i]], [0, y11[i], y22[i]])  # Pendulum 2
    return ln1, ln2

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
ani.save('two_pendula.gif', writer='pillow', fps=1/h)

plt.show()