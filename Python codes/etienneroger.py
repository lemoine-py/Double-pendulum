import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81

t_max = 20
h = 0.02 # Not smaller than that for the gif
N = int(np.floor(t_max/h))+1

w1_0 = 0
w2_0 = 0

th1_0 = np.pi + 0.1
th2_0 = np.pi

u0 = np.array([w1_0, w2_0, th1_0, th2_0])

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

def F_MIT(w1, w2, th1, th2): # MIT version of the derivative
    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
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
        k2 = h * f(u[i][0] + k1[0]/2, u[i][1] + k1[1]/2, u[i][2] + k1[2]/2, u[i][3] + k1[3]/2)
        k3 = h * f(u[i][0] + k2[0]/2, u[i][1] + k2[1]/2, u[i][2] + k2[2]/2, u[i][3] + k2[3]/2)
        k4 = h * f(u[i][0] + k3[0], u[i][1] + k3[1], u[i][2] + k3[2], u[i][3] + k3[3])
        u[i+1] = u[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, u

#t, u = solve_RK4(big_F, u0, h, t_max)

t1, u1 = solve_RK4(F_MIT, u0, h, t_max)

th1 = u1[:,2] 
th2 = u1[:,3]
w1 = u1[:,0]
w2 = u1[:,1]

### Energy calculation

T = np.zeros(N)
V = np.zeros(N)
E = np.zeros(N)

for i in range(N):
    T[i] = 0.5*m1*l1**2*w1[i]**2 + 0.5*m2*(l1**2*w1[i]**2 + l2**2*w2[i]**2 + 2*l1*l2*w1[i]*w2[i]*np.cos(th1[i]-th2[i])) # kinetic energy
    V[i] = -(m1 + m2)*g*l1*np.cos(th1[i]) - m2*g*l2*np.cos(th2[i]) # potential energy
    E[i] = T[i] + V[i] # total energy

# Plot the results
fig, ax = plt.subplots(2, 2)

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

plt.tight_layout()
plt.show()

#plt.plot(t, u[:,3])
plt.plot(th1, th2)  # brownian motion
plt.show()
plt.plot(t1, E) 
plt.show()

### Position in cartesian coordinates

x1 = np.zeros(N) # x component of m1
y1 = np.zeros(N) # y component of m1

x2 = np.zeros(N) # x component of m2
y2 = np.zeros(N) # y component of m2

for i in range(N):
    x1[i] = l1*np.sin(th1[i])
    y1[i] = -l1*np.cos(th1[i])
    
    x2[i] = x1[i] + l2*np.sin(th2[i])
    y2[i] = y1[i] - l2*np.cos(th2[i])

plt.plot(x1, y1)
plt.plot(x2, y2)

plt.show()

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    
fig, ax = plt.subplots(1,1, figsize=(4*(l1+l2),4*(l1+l2)))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ax.set_ylim(-2*(l1+l2),2*(l1+l2))
ax.set_xlim(-2*(l1+l2),2*(l1+l2))
ani = animation.FuncAnimation(fig, animate, frames=len(t1), interval=50)
ani.save('pen.gif',writer='pillow',fps=1/h)

# Jacobian
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

def jacobian(th1, th2, w1, w2):
    jac = np.zeros((4,4))
    
    jac[0][2] = 1 #dth1/dw1
    jac[1][3] = 1 #dth2/w2
    jac[2][2] = -2*np.sin(th1-th2)*m2*2*w1*l1*np.cos(th1-th2)/(l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[2][3] = -2*np.sin(th1-th2)*m2*2*w2*l2/(l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[3][2] = 2*np.sin(th1-th2)*2*w1*l1*(m1+m2)/(l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    jac[3][3] = 2*np.sin(th1-th2)*2*w2*l2*m2*np.cos(th1-th2)/(l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    
    num20 = (-g*(2*m1+m2)*np.cos(th1)-m2*g*np.cos(th1-2*th2)-2*np.cos(th1-th2)*m2*w2**2*l2-2*m2*np.cos(th1-th2)**2*w1**2*l1+2*np.sin(th1-th2)**2*m2*w1**2*l1)*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2)))*(l1*m2*2*np.sin(2*th1-2*th2))
    denom20 = (l1*(2*m2+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[2][0] = num20/denom20
    
    num21 = (2*m2*g*np.cos(th1-2*th2)+2*np.cos(th1-th2)*m2*w2**2*l2+2*np.cos(th1-th2)**2*m2*l2*w1**2-2*np.sin(th1-th2)**2*m2*w1**2*l1)*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2*w1**2*l2*np.cos(th1-th2)))*(-l1*m2*2*np.sin(2*th1-2*th2))
    denom21 = (l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[2][1] = num21/denom21
    
    num30 = 2*np.cos(th1-th2)*w1**2*l1*(m1+m2)+2*g*(m1+m2)*(np.cos(th1-th2)*np.cos(th1)-np.sin(th1-th2)*np.sin(th1)+2*w2**2*l2*m2*(np.cos(th1-th2)**2-np.sin(th1-th2)**2))*l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2)))*l1*(2*m2*np.sin(2*th1-2*th2))
    denom30 = (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[3][0] = num30/denom30
    
    num31 = -2*np.cos(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1))+2*w2**2*l2*m2*m2*(np.sin(th1-th2)**2-np.cos(th1-th2**2))*l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))-(2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2)))*(-2*l2*m2*np.sin(2*th1-2*th2))
    denom31 = (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))**2
    jac[3][1] = num31/denom31
                                                                                   
    return jac




def lyapunov(th):
    N = 60
    M = jacobian(th,th,0,0)
    dx_0 = [0, 10**(-10), 0, 0]
    t = np.linspace(0,N,N)
    dx = np.zeros((N,4))
    for i in range(N):
        dx[i] = sp.linalg.expm(M*t[i]) @ dx_0

    norm = np.zeros(N)
    lyap = np.zeros(N)
    for i in range(N):
        norm[i] = np.linalg.norm(dx[i])
        lyap[i] = np.log(norm[i]/norm[0])*1/t[i]
    return lyap[-1]
N = 100
M = jacobian(0.1,1,0,0)
dx_0 = [0, 0, 10**(-10), 0]
t = np.linspace(0,N,N)
dx = np.zeros((N,4))
for i in range(N):
    dx[i] = sp.linalg.expm(M*t[i]) @ dx_0

norm = np.zeros(N)
lyap = np.zeros(N)
for i in range(N):
    norm[i] = np.linalg.norm(dx[i])
    lyap[i] = np.log(norm[i]/norm[0])*1/t[i]
plt.plot(lyap)
eigenvalue, eigenvector = np.linalg.eig(M)
"""
N = 100
th = np.linspace(0,2*np.pi,N)
lyap = np.zeros(N)
for i in range (N):
    lyap[i] = lyapunov(th[i])
plt.plot(th,lyap)
"""




