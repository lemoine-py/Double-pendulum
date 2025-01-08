""" 
SPIN-UP phase for the double pendulum system

1. Jacobian function
2. RK4 for matrices
3. SPIN-UP phase -> 3.1 via RK4
                    3.2 via scipy.linalg.expm

!! Caution:
When the time step is too large, RK4 seems to produce a bias.
When the time step is small enough, the two methods give the same results.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tqdm # Progression bar

# math labels
lambda_math = r"$\lambda_{max}$"
delta_math = r"$\delta x_0$"
t_max_math = r"$t_{max}$"

# Parameters
l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81

### Defining the jacobian matrix ### -------------------------------------------------------------------------------------
def jacobian(w1, w2, th1, th2):
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

### Runge-Kutta 4 for matrices ###
def RK4_matrix(A, dx0, dt, N):
    """ Runge-Kutta 4 scheme for F = A@y """
    y = np.zeros((4, N+1))
    y[:, 0] = dx0 # delta's initial conditions
    for i in range(N):
        k1 = dt * (A @ y[:, i])
        k2 = dt * (A @ (y[:, i] + k1/2))
        k3 = dt * (A @ (y[:, i] + k2/2))
        k4 = dt * (A @ (y[:, i] + k3))
        y[:, i+1] = y[:, i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

### SPIN-UP phase ### -------------------------------------------------------------------------------------

# Arbitrary initial conditions
w1_0 = 0
w2_0 = 0
th1_0 = 3.14
th2_0 = 3.14

u0 = np.array([w1_0, w2_0, th1_0, th2_0])

# Time parameters
t_max = 100
dt = 0.01
N = int(np.floor(t_max/dt))+1
t_delta = np.linspace(0, t_max, N) # time array

theta_N = 30

# Arbitrary initial Delta_X
dx0 = [0, 0, 10**(-10), 0]

# Calling the jacobian function and fixing initial values
J = jacobian(w1_0, w2_0, th1_0, th2_0)

#-------------------------------------------
### Solving delta_X(t) with RK4
delta = RK4_matrix(J, dx0, dt, N)

def lyapunov_RK4(t, delta, N):
    lya = np.zeros(N)
    norm = np.zeros(N)
    for i in range (N):
        norm[i] = np.linalg.norm(delta[:,i])
    for i in range(N):
        lya[i] = np.log((norm[i])/norm[0])/t[i]
    return lya

lya_RK4 = lyapunov_RK4(t_delta, delta, N)
#-------------------------------------------
### Solving delta_X(t) with matrix exponentials
delta_a = np.zeros((N,4))

for i in range(N):
    delta_a[i] = sp.linalg.expm(J*t_delta[i]) @ dx0

def lyapunov_expm(t_delta, delta_a, N):
    lya = np.zeros(N)
    norm = np.zeros(N)
    for i in range (N):
        norm[i] = np.linalg.norm(delta_a[i])
    
    for i in range(N):
        lya[i] = np.log((norm[i])/norm[0])/t_delta[i]
    return lya

lya_expm = lyapunov_expm(t_delta, delta_a, N)

#-------------------------------------------
# Verifying with theoretical values of lyapunov_max #
eigenvalue, eigenvector = np.linalg.eig(J)
print()
print(f"Eigenvalues of the jacobian matrix when u0 = {u0}:")
print(eigenvalue)
print()
print("Theoretical lyapunov exponent:")
print(f"Lambda_max = {np.max(eigenvalue.real)}")
print()
print("Experimental lyapunov exponent:")
print(f"Lambda_max(t_max) = {lya_expm[-1]}")
print(f"Lambda_max(t=20) = {lya_expm[2000]}") # t = 20 if dt = 0.01
print(f"Lyapunov array average from t=20 to t_max: {np.average(lya_expm[2000:N])}")
print()
print(f"N.B. lyapunov array length = N = {len(lya_expm)}")
print()
#-------------------------------------------
# Plotting the single spin-up results

plt.figure()
plt.plot(t_delta, lya_RK4, label = "RK4")
plt.plot(t_delta, lya_expm, "--", label = "linalg.expm")
plt.plot(t_delta, np.max(eigenvalue.real)*np.ones(N), ":", color = "red", label = f"Theoretical {lambda_math} = {np.max(eigenvalue.real)}")
plt.suptitle(f"SPIN-UP for u0 = [{w1_0}, {w2_0}, {th1_0}, {th2_0}]")
plt.xlabel("Time (s)")
plt.ylabel("Lyapunov exponent")
plt.text(0.8, 0.3, f"dt = {dt} \n t_max = {t_max} \n {delta_math} = {dx0}", bbox = dict(facecolor = "white", alpha = 1), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.grid()
plt.legend()
plt.savefig(f"lyap_{t_max}_spinup_{th1_0}rad.png")