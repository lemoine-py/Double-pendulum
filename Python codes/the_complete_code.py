""" 
This code is the complete code for the Double Pendulum project.
By "complete", we mean that this code is self-sufficient and can be run as is.

For clarity, the repository is divided into several modules, that are copy-pasted here.

P.S. This code is actually dedicated to a certain collaborator, who will recognize himself.
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp # not used


# INITIALIZATION of the project and the parameters

def Initialize(planet):
    print("")
    print(" ___ DOUBLE PENDULUM SIMULATION ___ ")
    print("")
    print(" --- Initialization of the model ----")
    print("")
    g = planet()
    print("")
    l1 = float(input(" - Length l1 of the upper rod (m): ... "))
    print("")
    l2 = float(input(" - Length l2 of the lower rod (m): ... "))
    print("")
    m1 = float(input(" - Mass m1 of the upper rod (kg): ... "))
    print("")
    m2 = float(input(" - Mass m2 of the lower rod (kg): ... "))
    print("")
    theta1 = float(input(" - Initial angle theta_1 of the upper rod (degrees): ... "))
    print("")
    theta2 = float(input(" - Initial angle theta_2 of the lower rod (degrees): ... "))
    print("")
    omega1 = float(input(" - Initial angular velocity omega_1 of the upper rod (m/s): ... "))
    print("")
    omega2 = float(input(" - Initial angular velocity omega_2 of the lower rod (m/s): ... "))
    print("")
    return g, l1, l2, m1, m2, theta1, theta2, omega1, omega2

def planet():
    """ Sets the planet (and thus the gravitational acceleration g) where the pendulum is located """
    
    planet_list = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Sun"]
    
    planet = input("\n  ~ Choose the planet : \n - Mercury \n - Venus \n - Earth \n - Mars \n - Jupiter \n - Saturn \n - Uranus \n - Neptune \n - Pluto \n - Sun \n -> ... ")
    while planet not in planet_list:
        print("\n  ~ The planet you entered is not in the list. Please enter a valid planet.")
        planet = input("\n  ~ Choose the planet : \n - Mercury \n - Venus \n - Earth \n - Mars \n - Jupiter \n - Saturn \n - Uranus \n - Neptune \n - Pluto \n - Sun \n -> ... ")
    
    planet_g = {"Mercury" : 3.70,
                "Venus" : 8.87,
                "Earth" : 9.81,
                "Mars" : 3.71,
                "Jupiter" : 24.79,
                "Saturn" : 10.44,
                "Uranus" : 8.87,
                "Neptune" : 11.15,
                "Pluto" : 0.62,
                "Sun" : 274}

    if planet == "mercury" or planet == "Mercury":
        g = planet_g["Mercury"]
    elif planet == "venus" or planet == "Venus":
        g = planet_g["Venus"]
    elif planet == "earth" or planet == "Earth":
        g = planet_g["Earth"]
    elif planet == "mars" or planet == "Mars":
        g = planet_g["Mars"]
    elif planet == "jupiter" or planet == "Jupiter":
        g = planet_g["Jupiter"]
    elif planet == "saturn" or planet == "Saturn":
        g = planet_g["Saturn"]
    elif planet == "uranus" or planet == "Uranus":
        g = planet_g["Uranus"]
    elif planet == "neptune" or planet == "Neptune":
        g = planet_g["Neptune"]
    elif planet == "pluto" or planet == "Pluto":
        g = planet_g["Pluto"]
    elif planet == "sun" or planet == "Sun":
        g = planet_g["Sun"]
    else:
        print("Error")
        g = 0
    return g

# Call the initialization function to fix the parameters
g, l1, l2, m1, m2, th1_0, th2_0, w1_0, w2_0 = Initialize(planet)

# Time parameters
t_max = 50
h = 0.01 # Timestep size matters !!

# Initial conditions
u0 = np.array([w1_0, w2_0, th1_0, th2_0])


def big_F(w1, w2, th1, th2):
    """ Derivatives of the u array """
    
    num1 = -1*m2*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th1-th2))*np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom1 = ((m1+m2)*l1-l1*np.cos(th1-th2)**2*m2)
    
    num2 = -1*(m1+m2)*(l1*w1**2*np.sin(th1-th2)-g*np.sin(th2))/np.cos(th1-th2)-m2*l2*w2**2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    denom2 = -1*l2*(m1+m2)/np.cos(th1-th2)+m2*l2*np.cos(th1-th2)
    
    w1_dot = num1/denom1
    w2_dot = num2/denom2
    
    return np.array([w1_dot, w2_dot, w1, w2])

def solve_RK4(f, u0, h, t_max):
    """ Solve the system of ODEs using the Runge-Kutta 4 method """
    
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
ax[0, 0].set_xlabel("Time (s)")
    
ax[0, 1].plot(t, u[:,1])
ax[0, 1].set_xlabel("Time (s)")
    
ax[1, 0].plot(t, u[:,2])
ax[1, 0].set_xlabel("Time (s)")
    
ax[1, 1].plot(t, u[:,3])
ax[1, 1].set_xlabel("Time (s)")

plt.tight_layout()