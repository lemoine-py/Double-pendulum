"""
1. Colormap
Calculates the local Lyapunov exponents for each pair of initial angles
(theta_1, theta_2) in the range [0, pi] x [0, pi] and plots the results as a colormap.

By default it is set to only use the theoretical way of calculating the Lyapunov exponents,
which is by calculating the maximum eigenvalue of the Jacobian matrix at each point.
If you want to calculate the Lyapunov exponents using the matrix exponential, you can
uncomment the lines in the for loop and comment the theoretical way.

2. Simple 2D plot
Calculates the local Lyapunov exponents for each initial condition where theta_1 = theta_2 = theta.
It then plots 
a. the theoretical maximum Lyapunov exponent,
b. the "matrix exponentials" Lyapunov exponent at t = 20 and t = 100,
and their average over time.

"""

# Libraries
import numpy as np
import scipy as sp # For the matrix exponential
import matplotlib.pyplot as plt
from tqdm import tqdm # Progression bar

# math labels
lambda_math = r"$\lambda_{max}$"
lambda_n_math = r"$\lambda_{max,n}$"
delta_math = r"$\delta x_0$"
t_max_math = r"$t_{max}$"
t_step_math = r"$t_{step}$"

# Parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

# Time parameters
theta_N = 314 
dt = 0.01
t_max = 100
N = int(np.floor(t_max/dt))+1
t_delta = np.linspace(0, t_max, N)

# Initial dx0
dx0 = np.array([0, 0, 1, 0])
dx0 = dx0/(np.linalg.norm(dx0)*10**10) # so that norm(dx0) = 10**(-10)

def jacobian(w1, w2, th1, th2):
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

def lyapunov_expm(t_delta, delta_a, N):
    lya = np.zeros(N)
    norm = np.zeros(N)
    for i in range (N):
        norm[i] = np.linalg.norm(delta_a[i])
    
    for i in range(N):
        lya[i] = np.log((norm[i])/norm[0])/t_delta[i]
    return lya

print()
print(f"t_max = {t_max}")
print(f"dt = {dt}")
print(f"theta_N = {theta_N}")
print()

th = np.linspace(-np.pi,np.pi,2*theta_N) # Angle array
local_lyap_pq = np.zeros((2*theta_N,2*theta_N))

with tqdm(total=2*theta_N*2*theta_N*N) as pbar: # Progression bar
    for p in range(2*theta_N):
        for q in range(2*theta_N):
            Jpq = jacobian(0, 0, th[p], th[q])

            ## Theoretical way:
            local_lyap_pq[p][q] = np.max(np.linalg.eig(Jpq)[0].real)
            pbar.update(N)

            ## Matrix exponential way:
            # delta_ap = np.zeros((N,4))
            # for i in range(N):
            #     delta_ap[i] = sp.linalg.expm(Jpq*t_delta[i]) @ dx0
            #     pbar.update(1)
            # lya_ap = lyapunov_expm(t_delta, delta_ap, N) 

            # filtered_lya_ap = np.array(lya_ap)[np.isfinite(lya_ap)] # Filtering the NaN and inf values
            # local_lyap_pq[p][q] = np.average(filtered_lya_ap)

# Plot the array as a colormap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(local_lyap_pq, cmap='jet', extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower')

ax.set_xlabel(r'$\theta_2$')
ax.set_ylabel(r'$\theta_1$')

cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Lyapunov exponent')

plt.title(r'Local $\lambda_{max}$ for each ($\theta_1$,$\theta_2$), starting from $\delta x_0$ = [0,0,1,0]e-10')
plt.savefig("local_lyap_cmap_z.png")


#----------------------------------------------------------------------------------------------


# SPIN-UP for each initial conditions where theta1 = theta2 = theta
th = np.linspace(0,np.pi,theta_N) # Angle array

theta_theta = [np.array([0, 0, th[p], th[p]])  for p in range(theta_N)]
theta_zero = [np.array([0, 0, th[p], 0]) for p in range(theta_N)]
zero_theta = [np.array([0, 0, 0, th[p]]) for p in range(theta_N)]

lyap_spinup1 = np.zeros(theta_N)
lyap_spinup2 = np.zeros(theta_N)
lyap_spinup_ave = np.zeros(theta_N)
lyap_spinup_theo = np.zeros(theta_N)

with tqdm(total=theta_N*N) as pbar: # Progression bar
    for p in range(theta_N):
        Jp = jacobian(0, 0, th[p], th[p])
        delta_ap = np.zeros((N,4))
        for i in range(N):
            delta_ap[i] = sp.linalg.expm(Jp*t_delta[i]) @ dx0
            pbar.update(1)
        lya_ap = lyapunov_expm(t_delta, delta_ap, N) 

        lyap_spinup_theo[p] = np.max(np.linalg.eig(Jp)[0].real)
        lyap_spinup1[p] = lya_ap[2000]
        lyap_spinup2[p] = lya_ap[10000]
        filtered_lya_ap = np.array(lya_ap)[np.isfinite(lya_ap)] # Filtering the NaN and inf values
        lyap_spinup_ave[p] = np.average(filtered_lya_ap)
  
# Plotting the SPIN-UP for each initial conditions
plt.figure()
plt.plot(th, lyap_spinup_theo, color = "red", label = f"Theoretical {lambda_math}")
plt.plot(th, lyap_spinup1, "--", color = "limegreen", label = f"{lambda_math} at t = 20")
plt.plot(th, lyap_spinup2, "--", color = "blue", label = f"{lambda_math} at t = 100")
plt.plot(th, lyap_spinup_ave, color = "orange", label = f"np.average(filtered_{lambda_math}(t))")


plt.suptitle(r"SPIN-UP for each initial conditions where $\theta_{1,0} = \theta_{2,0} = \theta$")
plt.ylabel("Lyapunov exponent")
plt.xlabel(r"$\theta$ (rad)")
plt.text(0.2, 0.3, f"dt = {dt}\n {t_max_math} = {t_max}\n {delta_math} = [0,0,1,0]e-10", bbox = dict(facecolor = "white", alpha = 1), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.legend()
plt.grid()

plt.savefig("local_lyap_theta_theta.png")

plt.show()