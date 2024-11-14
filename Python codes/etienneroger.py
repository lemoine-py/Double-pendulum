
"""
A certain collaborator's personal code.
"""

import numpy as np
import matplotlib.pyplot as plt

l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81

t_max = 50
h = 0.01

w1_0 = 0
w2_0 = 0

th1_0 = -0.01
th2_0 = 0.01

u0 = np.array([w1_0, w2_0, th1_0, th2_0])

def big_F(w1, w2, th1, th2):
    num1 = -1*m2*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th1-th2))*np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom1 = ((m1+m2)*l1-l1*np.cos(th1-th2)**2*m2)
    
    num2 = -1*(m1+m2)*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th2))/np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom2 = -1*l2*(m1+m2)/np.cos(th1-th2)+m2*l2*np.cos(th1-th2)
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2
    
    return np.array([w1_dot, w2_dot, w1, w2])

def solve_RK4(f, u0, h, t_max):
    N = int(np.floor(t_max / h)) + 1
    t = np.linspace(0, t_max, N)
    u = np.zeros((N,4))
    u[0] = u0
    for i in range(N - 1):
        k1 = h * f(u[i][0], u[i][1], u[i][2], u[i][3])
        k2 = h * f(u[i][0] + k1[0], u[i][1] + k1[1], u[i][2] + k1[2], u[i][3] + k1[3])
        k3 = h * f(u[i][0] + k2[0], u[i][1] + k2[1], u[i][2] + k2[2], u[i][3] + k2[3])
        k4 = h * f(u[i][0] + k3[0], u[i][1] + k3[1], u[i][2] + k3[2], u[i][3] + k3[3])
        u[i+1] = u[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, u

t, u = solve_RK4(big_F, u0, h, t_max)


# Plot the results
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(t, u[:,0])
ax[0, 0].set_title("w1")
    
ax[0, 1].plot(t, u[:,1])
ax[0, 1].set_title("w2")
    
ax[1, 0].plot(t, u[:,2])
ax[1, 0].set_title("th1")
    
ax[1, 1].plot(t, u[:,3])
ax[1, 1].set_title("th2")

plt.tight_layout()
plt.show()