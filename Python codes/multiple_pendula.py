""" 
This code aims to compare two or more pendula 
with different initial conditions and plot the results. 
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
from Initialize_planet import Initialize, planet

# Lets the user choose the parameters and initial conditions by a series of prompts
g, l1, l2, m1, m2, th1_0, th2_0, w1_0, w2_0 = Initialize(planet)
"""

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

### Initial conditions
# Pendulum 1
w1_0 = 0
w2_0 = 0
th1_0 = np.pi + 0.1
th2_0 = np.pi
u0 = np.array([w1_0, w2_0, th1_0, th2_0])
# Pendulum 2
ww1_0 = 0
ww2_0 = 0
thth1_0 = np.pi + 0.2
thth2_0 = np.pi
uu0 = np.array([ww1_0, ww2_0, thth1_0, thth2_0])

### Defining functions to solve the differential equation
def big_F(w1, w2, th1, th2):
    """ Derivatives of the u array """
    
    # Separating the numerator and the denominator of the equations for clarity
    num1 = -1*m2*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th1-th2))*np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom1 = ((m1+m2)*l1-l1*np.cos(th1-th2)**2*m2)
    
    num2 = -1*(m1+m2)*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th2))/np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom2 = -1*l2*(m1+m2)/np.cos(th1-th2)+m2*l2*np.cos(th1-th2)
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2
    
    return np.array([w1_dot, w2_dot, w1, w2])

def F_MIT(w1, w2, th1, th2): 
    """MIT version of the derivative"""

    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2    
    
    return np.array([w1_dot, w2_dot, w1, w2])

def solve_RK4(f, u0, h, t_max):
    """ Solves the differential equation using the Runge-Kutta 4th order method """

    N = int(np.floor(t_max / h)) + 1 # Redundant definition of the already globally defined N
    t = np.linspace(0, t_max, N)
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
#t, u = solve_RK4(big_F, u0, h, t_max)
t1, u1 = solve_RK4(F_MIT, u0, h, t_max)
tt1, uu1 = solve_RK4(F_MIT, uu0, h, t_max)
# Pendulum 1
th1 = u1[:,2] 
th2 = u1[:,3]
w1 = u1[:,0]
w2 = u1[:,1]
# Pendulum 2
thth1 = uu1[:,2]
thth2 = uu1[:,3]
ww1 = uu1[:,0]
ww2 = uu1[:,1]

sol = [[th1, th2, w1, w2], [thth1, thth2, ww1, ww2]]

### Energy calculations
T = np.zeros(N)
V = np.zeros(N)
E = np.zeros(N)

def energy(m1, m2, l1, l2, g, w1, w2, th1, th2, T, V, E):
    for i in range(N):
        T[i] = 0.5*m1*l1**2*w1[i]**2 + 0.5*m2*(l1**2*w1[i]**2 + l2**2*w2[i]**2 + 2*l1*l2*w1[i]*w2[i]*np.cos(th1[i]-th2[i])) # kinetic energy
        V[i] = -(m1 + m2)*g*l1*np.cos(th1[i]) - m2*g*l2*np.cos(th2[i]) # potential energy
        E[i] = T[i] + V[i] # total energy
    return T, V, E

T1, V1, E1 = energy(m1, m2, l1, l2, g, w1, w2, th1, th2, T, V, E)
T2, V2, E2 = energy(m1, m2, l1, l2, g, ww1, ww2, thth1, thth2, T, V, E)


# Plot the results
fig, ax = plt.subplots(2, 4)

ax[0, 0].plot(t1, w1)
ax[0, 0].set_xlabel("Time (s)")
ax[0, 0].set_title("w1")
ax[0, 0].grid()
    
ax[0, 1].plot(t1, w2)
ax[0, 1].set_xlabel("Time (s)")
ax[0, 1].set_title("w2")
ax[0, 1].grid()
    
ax[1, 0].plot(t1, th1)
ax[1, 0].set_xlabel("Time (s)")
ax[1, 0].set_title("theta1")
ax[1, 0].grid()
    
ax[1, 1].plot(t1, th2)
ax[1, 1].set_xlabel("Time (s)")
ax[1, 1].set_title("theta2")
ax[1, 1].grid()

ax[0, 2].plot(tt1, ww1)
ax[0, 2].set_xlabel("Time (s)")
ax[0, 2].set_title("ww1")
ax[0, 2].grid()

ax[0, 3].plot(tt1, ww2)
ax[0, 3].set_xlabel("Time (s)")
ax[0, 3].set_title("ww2")
ax[0, 3].grid()

ax[1, 2].plot(tt1, thth1)
ax[1, 2].set_xlabel("Time (s)")
ax[1, 2].set_title("thth1")
ax[1, 2].grid()

ax[1, 3].plot(tt1, thth2)
ax[1, 3].set_xlabel("Time (s)")
ax[1, 3].set_title("thth2")
ax[1, 3].grid()

plt.tight_layout()
#plt.savefig("multiple_angles_and_velocities.png")

plt.figure()
plt.plot(th1, th2)  # brownian motion (theta1 vs theta2)
plt.plot(thth1, thth2)
plt.suptitle("Brownian motion of the double pendulum")
plt.xlabel("Theta 1")
plt.ylabel("Theta 2")
plt.grid()
#plt.savefig("multiple_brownian_motion.png")

plt.figure()
plt.plot(t1, E) # total energy vs time
plt.plot(tt1, E2)
plt.suptitle("Total energy of the double pendulum")
plt.xlabel("Time (s)")
plt.ylabel("Total energy (J)")
plt.grid()
#plt.savefig("multiple_total_energy.png")

### Position in cartesian coordinates

x1 = np.zeros(N) # x component of m1
y1 = np.zeros(N) # y component of m1

x2 = np.zeros(N) # x component of m2
y2 = np.zeros(N) # y component of m2

xx1 = np.zeros(N) # x component of m11
yy1 = np.zeros(N) # y component of m11

xx2 = np.zeros(N) # x component of m22
yy2 = np.zeros(N) # y component of m22

def cartesian(th1, th2, l1, l2):
    for i in range(N):
        x1[i] = l1*np.sin(th1[i])
        y1[i] = -l1*np.cos(th1[i])
        
        x2[i] = x1[i] + l2*np.sin(th2[i])
        y2[i] = y1[i] - l2*np.cos(th2[i])
    return x1, y1, x2, y2

x1, y1, x2, y2 = cartesian(th1, th2, l1, l2)
xx1, yy1, xx2, yy2 = cartesian(thth1, thth2, l1, l2)

figa, axa = plt.subplots(2,1)
axa[0,0].plot(x1, y1, label="m1")
axa[0,0].plot(x2, y2, label="m2")
axa[0,0].xlabel("x")
axa[0,0].ylabel("y")
axa[0, 0].set_title("Pendulum 1")

axa[1,0].plot(xx1, yy1, label="m11")
axa[1,0].plot(xx2, yy2, label="m22")
axa[1,0].xlabel("x")
axa[1,0].ylabel("y")
axa[1, 0].set_title("Pendulum 2")

plt.legend()
#plt.savefig("XY_position.png")

plt.show() # Shows at once every plot that has been produced before (in multiple windows)

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    ln2.set_data([0, xx1[i], xx2[i]], [0, yy1[i], yy2[i]])
    
fig, ax = plt.subplots(1,1, figsize=(4*(l1+l2),4*(l1+l2)))
ax.set_facecolor('k')
#ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
#ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = ax.plot([], [], 'o-', lw=3, markersize=8, color = "red", label = "Pendulum 1")
ln2, = ax.plot([], [], 'o-', lw=3, markersize=8, color = "blue", label = "Pendulum 2")
ax.set_ylim(-1.5*(l1+l2),1.5*(l1+l2))
ax.set_xlim(-1.5*(l1+l2),1.5*(l1+l2))
ani = animation.FuncAnimation(fig, animate, frames=len(t1), interval=50)
ani.save('pen.gif',writer='pillow',fps=1/h)