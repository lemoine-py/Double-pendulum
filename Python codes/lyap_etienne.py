import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

# Time parameters
t_max = 20
h = 0.01 
N = int(np.floor(t_max/h))+1 # about 2000 timesteps

# Initial conditions
w1_0 = 0
w2_0 = 0
th1_0 = 1.5
th2_0 = 0
u0 = np.array([w1_0, w2_0, th1_0, th2_0])

# Initial delta_x
#dx0 = np.random.rand(4)
dx0 = np.array([0, 0, 1, 0])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # norm(dx0) = 10**(-10)
u0d = u0 + dx0

def F_MIT(w1, w2, th1, th2): # MIT version of the derivative
    num1 = -g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*np.cos(th1-th2))
    denom1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    num2 = 2*np.sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+w2**2*l2*m2*np.cos(th1-th2))
    denom2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2    
    
    return np.array([w1_dot, w2_dot, w1, w2])
    
    
def solve_RK4(f, u0, h, t_max):
    """ Returns : t (time array), u (4x1 array of last computed values)"""
    N = int(np.floor(t_max / h)) + 1
    t = np.linspace(0, t_max, N)
    u = np.zeros(4) # u does not store each set of 4 values for each timestep ! Only the last computed set
    u = u0
    for i in range(N - 1):
        k1 = h * f(u[0], u[1], u[2], u[3])
        k2 = h * f(u[0] + k1[0]/2, u[1] + k1[1]/2, u[2] + k1[2]/2, u[3] + k1[3]/2)
        k3 = h * f(u[0] + k2[0]/2, u[1] + k2[1]/2, u[2] + k2[2]/2, u[3] + k2[3]/2)
        k4 = h * f(u[0] + k3[0], u[1] + k3[1], u[2] + k3[2], u[3] + k3[3])
        u = u + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, u

t1, u1 = solve_RK4(F_MIT, u0, h, t_max)
t2, u1d = solve_RK4(F_MIT, u0d, h, t_max)
"""
th1 = u1[:,2] 
th2 = u1[:,3]
w1 = u1[:,0]
w2 = u1[:,1]

th1d = u1d[:,2] 
th2d = u1d[:,3]
w1d = u1d[:,0]
w2d = u1d[:,1]
"""

def jacobian(th1, th2, w1, w2):
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


def lyapunov(u, dx):
    """ Computing the lyapunov exponent at t_max starting from a given u0 and delta_x"""
    t = 20 # = t_max = 20 sec
    M = jacobian(u[2],u[3],u[0],u[1]) # evaluating the jacobian at the 20th timestep
    dxf = sp.linalg.expm(M*t) @ dx # solving for delta_x
    norm = np.linalg.norm(dxf)
    lyapf = np.log(norm/np.linalg.norm(dx))*1/N # Lyapunov exponent of the 20th timestep
    return lyapf, dxf
"""
# SPIN-UP
t_max = 100
N = 100
th1_0 = 1.5
M = jacobian(th1_0,0,0,0)
dx_0 = [0, 0, 10**(-10), 0]
t = np.linspace(0,t_max,N)
dx = np.zeros((N,4))
for i in range(N):
    dx[i] = sp.linalg.expm(M*t[i]) @ dx_0

norm = np.zeros(N)
lyap = np.zeros(N)
for i in range(N):
    norm[i] = np.linalg.norm(dx[i])
    lyap[i] = np.log(norm[i]/norm[0])*1/t[i]

plt.figure()
plt.plot(t,lyap, label = f"lyap[-1] = {lyap[-1]}")
plt.suptitle(f"SPIN-UP for theta1 = {th1_0}")
plt.ylabel("Lyapunov exponent")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()
plt.show()

# Verifying with theoretical values of lyapunov_max #
eigenvalue, eigenvector = np.linalg.eig(M)
print()
print(eigenvalue)
print(f"lyap[-1] = {lyap[-1]}")
print()
"""

#-------------------------------------------
### Theta-theta initial conditions ###
th = np.linspace(0,np.pi,30) # Angle array
lya_th = np.zeros(30) # "Lyapunov for each angle" array

for p in range(30):
    u0 = np.array([0, 0, th[p], th[p]])
    lya = np.zeros(50)
    for i in range(50):
        t1, u1 = solve_RK4(F_MIT, u0, h, t_max) # updating th1, th2, w1, w2
        lya[i], dx = lyapunov(u0, dx0)
        u0 = u1 # updating initial conditions
        dx0 = dx/(np.linalg.norm(dx)*10**10) # Renormalising so that norm(dx0) = 10**(-10)
    lya_th[p] = np.average(lya)

plt.figure()
plt.plot(th, lya_th, label = "Lyapunov exponent average")
plt.suptitle("Theta-theta initial conditions")
plt.ylabel("Lyapunov")
plt.xlabel("Theta (rad)")
plt.legend()
plt.grid()
plt.show()

#-------------------------------------------

dx0 = np.array([0, 0, 1, 0])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # norm(dx0) = 10**(-10)
### Theta-0 initial conditions ###
th = np.linspace(0,np.pi,30) # Angle array
lya_th = np.zeros(30) # "Lyapunov for each angle" array

for p in range(30):
    u0 = np.array([0, 0, th[p], 0])
    lya = np.zeros(50)
    for i in range(50):
        t1, u1 = solve_RK4(F_MIT, u0, h, t_max) # updating th1, th2, w1, w2
        lya[i], dx = lyapunov(u0, dx0)
        u0 = u1 # updating initial conditions
        dx0 = dx/(np.linalg.norm(dx)*10**10) # Renormalising so that norm(dx0) = 10**(-10)
    lya_th[p] = np.average(lya)

plt.figure()
plt.plot(th, lya_th, label = "Lyapunov exponent average")
plt.suptitle("Theta-0 initial conditions")
plt.ylabel("Lyapunov")
plt.xlabel("Theta (rad)")
plt.legend()
plt.grid()
plt.show()

#-------------------------------------------
dx0 = np.array([0, 0, 1, 0])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # norm(dx0) = 10**(-10)
### 0-Theta initial conditions ###
th = np.linspace(0,np.pi,30) # Angle array
lya_th = np.zeros(30) # "Lyapunov for each angle" array

for p in range(30):
    u0 = np.array([0, 0, 0, th[p]])
    lya = np.zeros(50)
    for i in range(50):
        t1, u1 = solve_RK4(F_MIT, u0, h, t_max) # updating th1, th2, w1, w2
        lya[i], dx = lyapunov(u0, dx0)
        u0 = u1 # updating initial conditions
        dx0 = dx/(np.linalg.norm(dx)*10**10) # Renormalising so that norm(dx0) = 10**(-10)
    lya_th[p] = np.average(lya)

plt.figure()
plt.plot(th, lya_th, label = "Lyapunov exponent average")
plt.suptitle("0-Theta initial conditions")
plt.ylabel("Lyapunov")
plt.xlabel("Theta (rad)")
plt.legend()
plt.grid()
plt.show()

