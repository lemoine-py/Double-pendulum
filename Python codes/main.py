import matplotlib.pyplot as plt
import numpy as np

from Initialize_planet import Initialize, planet

# Parameters
g, l, m, theta1, theta2, omega1, omega2 = Initialize(planet)


def RK4(t, y, h, f):  
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)    

    return 1/6*(k1 + 2 * k2 + 2 * k3 + k4) 


def big_F(t, y, g, l, m, theta1, theta2, omega1, omega2):
    
    w1 = 1
    w2 = 1

    # System of 4 first order ODEs
    f0 = w1 # d/dt th1 = w1
    f1 = w2 # d/dt th2 = w2

    f2 = # d/dt w1 = d2/dt2 th1 = ...
    
    f3 = # d/dt w2 = d2/dt2 th2 = ...
    
    return np.array([f0, f1, f2, f3])


def main():
    t_0 = 0
    T = 100
    n = 2048
    # Initial conditions
    y_0 = [np.pi/4, np.pi/4, 0, 0]                  
    
    # Domain discretization
    t_n = np.linspace(T) 
    y_n = [np.array(y_0)]  
    
    #Step size
    h  = (T - t0)/n        
    
    while t_n[-1] < T:
        y_n.append(y_n[-1] + h * RK4(t_n[-1], y_n[-1], h, RHS))
        t_n.append(t_n[-1] + h)
    
    
    print(y_n)
 

main()