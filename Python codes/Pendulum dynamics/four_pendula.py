""" 
This code aims to compare four pendula with different initial conditions.
It simulates the dynamics of four double pendulum systems and plots the results.

This code produces 6 files : 

1. <angles1_velocities1.png> : four subplots for each angles and angular velocities with respect to time
2. <angles2_velocities2.png> : same as 1. but for another double pendulum
3. <four_parametric.png> : theta_1 vs theta_2 plot, illustrates the brownian motion of the system
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
t_max = 100
dt = 0.02 # Not smaller than that for the gif
N = int(np.floor(t_max/dt))+1
t = np.linspace(0, t_max, N)

### Initial conditions
# Initial angles templates
critical = 1.38 # just below critical angle by about 0.002
big_critical = 1.3 #1.4 + 0.1 was the former value
high = 2.8 # high energy
low = 0.1 # low energy
# Angle offset aka delta
big = 0.1
medium = 0.01
small = 0.001
# Choosing between templates
angle = low #change at will
delta = big #change at will
thetas = [angle + i*delta for i in range(4)]



# Pendulum A
w1_0 = 0
w2_0 = 0
th1_0 = thetas[0]
th2_0 = thetas[0]
u0 = np.array([w1_0, w2_0, th1_0, th2_0])
# Pendulum B
w11_0 = 0
w22_0 = 0
th11_0 = thetas[1]
th22_0 = thetas[1]
u00 = np.array([w11_0, w22_0, th11_0, th22_0])
# Pendulum C
w111_0 = 0
w222_0 = 0
th111_0 = thetas[2]
th222_0 = thetas[2]
u000 = np.array([w111_0, w222_0, th111_0, th222_0])
# Pendulum D
w1111_0 = 0
w2222_0 = 0
th1111_0 = thetas[3]
th2222_0 = thetas[3]
u0000 = np.array([w1111_0, w2222_0, th1111_0, th2222_0])

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

def solve_RK4(f, u0, dt, N):
    """ Solves the differential equation using the Runge-Kutta 4th order method """
    u = np.zeros((N,4))
    u[0] = u0
    for i in range(N - 1):
        k1 = dt * f(u[i][0], u[i][1], u[i][2], u[i][3])
        k2 = dt * f(u[i][0] + k1[0]/2, u[i][1] + k1[1]/2, u[i][2] + k1[2]/2, u[i][3] + k1[3]/2)
        k3 = dt * f(u[i][0] + k2[0]/2, u[i][1] + k2[1]/2, u[i][2] + k2[2]/2, u[i][3] + k2[3]/2)
        k4 = dt * f(u[i][0] + k3[0], u[i][1] + k3[1], u[i][2] + k3[2], u[i][3] + k3[3])
        u[i+1] = u[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u

### Storing the solutions in arrays u1, u11, u111, u1111
u1 = solve_RK4(F_deriv, u0, dt, N)
u11 = solve_RK4(F_deriv, u00, dt, N)
u111 = solve_RK4(F_deriv, u000, dt, N)
u1111 = solve_RK4(F_deriv, u0000, dt, N)

# Pendulum A
th1 = u1[:,2] 
th2 = u1[:,3]
w1 = u1[:,0]
w2 = u1[:,1]
# Pendulum B
th11 = u11[:,2]
th22 = u11[:,3]
w11 = u11[:,0]
w22 = u11[:,1]
# Pendulum C
th111 = u111[:,2]
th222 = u111[:,3]
w111 = u111[:,0]
w222 = u111[:,1]
# Pendulum D
th1111 = u1111[:,2]
th2222 = u1111[:,3]
w1111 = u1111[:,0]
w2222 = u1111[:,1]

### ---------------------------------------------------------------------------------------------------------
# Plotting the solutions i.e. all 4 components of u
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(t, w1, color = "magenta")
ax[0, 0].set_xlabel("Time (s)")
ax[0,0].set_ylabel("Angular velocity (rad/s)")
ax[0, 0].set_title(r"$\omega_{A1}$")
ax[0, 0].grid()
    
ax[0, 1].plot(t, w2, color = "magenta")
ax[0, 1].set_xlabel("Time (s)")
ax[0, 1].set_title(r"$\omega_{A2}$")
ax[0, 1].grid()
    
ax[1, 0].plot(t, th1, color = "magenta")
ax[1, 0].set_xlabel("Time (s)")
ax[1,0].set_ylabel("Angle (rad)")
ax[1, 0].set_title(r"$\theta_{A1}$")
ax[1, 0].grid()
    
ax[1, 1].plot(t, th2, color = "magenta")
ax[1, 1].set_xlabel("Time (s)")
ax[1, 1].set_title(r"$\theta_{A2}$")
ax[1, 1].grid()

fig.suptitle(r"Angles and angular velocities, $\theta_0$  = %.3f rad" % angle)
fig.tight_layout()
fig.savefig("angles1_velocities1.png")

figg, axx = plt.subplots(2,2)

axx[0, 0].plot(t, w1111, color = "mediumpurple")
axx[0, 0].set_xlabel("Time (s)")
axx[0, 0].set_ylabel("Angular velocity (rad/s)")
axx[0, 0].set_title(r"$\omega_{D1}$")
axx[0, 0].grid()

axx[0, 1].plot(t, w2222, color = "mediumpurple")
axx[0, 1].set_xlabel("Time (s)")
axx[0, 1].set_title(r"$\omega_{D2}$")
axx[0, 1].grid()

axx[1, 0].plot(t, th1111, color = "mediumpurple")
axx[1, 0].set_xlabel("Time (s)")
axx[1, 0].set_ylabel("Angle (rad)")
axx[1, 0].set_title(r"$\theta_{D1}$")
axx[1, 0].grid()

axx[1, 1].plot(t, th2222, color = "mediumpurple")
axx[1, 1].set_xlabel("Time (s)")
axx[1, 1].set_title(r"$\theta_{D2}$")
axx[1, 1].grid()

figg.suptitle(r"Angles and angular velocities, $\theta_0$  = %.3f rad" % (angle+3*delta))
figg.tight_layout()
figg.savefig("angles2_velocities2.png")

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
T3, V3, E3 = energy(m1, m2, l1, l2, g, w111, w222, th111, th222)
T4, V4, E4 = energy(m1, m2, l1, l2, g, w1111, w2222, th1111, th2222)

# Plotting the difference between the initial total energy and the computed total energy along time
plt.figure()
plt.plot(t, E1[0]-E1, label = "Pendulum A", color = "magenta")
plt.plot(t, E2[0]-E2, label = "Pendulum B", color = "cyan")
plt.plot(t, E3[0]-E3, label = "Pendulum C", color = "gold")
plt.plot(t, E4[0]-E4, label = "Pendulum D", color = "gray")

plt.suptitle("Total energy losses due to RK4 inacurracy")
plt.xlabel("Time [s]")
plt.ylabel("E(t=0) - E(t) [J]")
plt.text(0.2, 0.6, f"Initial energy : %.2f J \n t_max = {t_max} s \n theta_0 = {angle} rad \n delta = {delta} rad" % E1[0], bbox = dict(facecolor = "white", alpha = 1), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.legend()
plt.grid()
plt.savefig("four_energy_loss.png")

### Brownian Motion plot (th1 vs th2) ### -------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(th1111, th2222, label = "Pendulum D", color = "grey")
ax.plot(th111, th222, label = "Pendulum C", color = "gold")
ax.plot(th11, th22, label = "Pendulum B", color = "cyan")
ax.plot(th1, th2, label = "Pendulum A", color = "magenta")
ax.plot(th1[0], th2[0], "-o", label = "Starting point", color = "red")
ax.set_title(f"t_max = {t_max} s, theta_0 = {angle} rad, delta = {delta} rad")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.legend()
ax.grid()
fig.savefig("four_parametric.png")


### Positions in cartesian coordinates ### -------------------------------------------------------------------

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
x111, y111, x222, y222 = cartesian(th111, th222, l1, l2)
x1111, y1111, x2222, y2222 = cartesian(th1111, th2222, l1, l2)

# Plot the paths of the pendula
figa, axa = plt.subplots(2,2)
axa[0,0].plot(x1, y1, label="m1")
axa[0,0].plot(x2, y2, label="m2", color = "magenta")
axa[0,0].set_xlabel("x")
axa[0,0].set_ylabel("y")
axa[0,0].set_title("Pendulum A")
axa[0,0].legend()
axa[0,0].set_aspect('equal')

axa[0,1].plot(x11, y11, label="m1")
axa[0,1].plot(x22, y22, label="m2", color = "cyan")
axa[0,1].set_xlabel("x")
axa[0,1].set_ylabel("y")
axa[0,1].set_title("Pendulum B")
axa[0,1].legend()
axa[0,1].set_aspect('equal')

axa[1,0].plot(x111, y111, label="m1")
axa[1,0].plot(x222, y222, label="m2", color = "gold")
axa[1,0].set_xlabel("x")
axa[1,0].set_ylabel("y")
axa[1,0].set_title("Pendulum C")
axa[1,0].legend()
axa[1,0].set_aspect('equal')

axa[1,1].plot(x1111, y1111, label="m1")
axa[1,1].plot(x2222, y2222, label="m2", color = "grey")
axa[1,1].set_xlabel("x")
axa[1,1].set_ylabel("y")
axa[1,1].set_title("Pendulum D")
axa[1,1].legend()
axa[1,1].set_aspect('equal')

figa.suptitle(f"t_max = {t_max} s, theta_0 = {angle} rad, delta = {delta} rad")
figa.tight_layout()
figa.savefig("four_XY_paths.png")

plt.show() # Shows at once every plot that has been produced before (in multiple windows)


### ANIMATION GIF ### ---------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3 * (l1 + l2), 3 * (l1 + l2)))
ax.set_facecolor('k')
ax.set_xlim(-1.5 * (l1 + l2), 1.5 * (l1 + l2))
ax.set_ylim(-1.5 * (l1 + l2), 1.5 * (l1 + l2))
#ax.plot(0, 0, 'o', color = "green")  # Origin

# Initialization of the pendulums
ln1, = ax.plot([], [], 'o-', lw=3, markersize=8, color="magenta", label="Pendulum A")
ln2, = ax.plot([], [], 'o-', lw=3, markersize=8, color="cyan", label="Pendulum B")
ln3, = ax.plot([], [], 'o-', lw=3, markersize=8, color="gold", label="Pendulum C") 
ln4, = ax.plot([], [], 'o-', lw=3, markersize=8, color="white", label="Pendulum D")
ax.legend()

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])  # Pendulum A
    ln2.set_data([0, x11[i], x22[i]], [0, y11[i], y22[i]])  # Pendulum B
    ln3.set_data([0, x111[i], x222[i]], [0, y111[i], y222[i]])  # Pendulum C
    ln4.set_data([0, x1111[i], x2222[i]], [0, y1111[i], y2222[i]])  # Pendulum D
    return ln1, ln2

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
ani.save('four_pendula.gif', writer='pillow', fps=1/dt) # Save the animation as a gif