import numpy as np
import scipy as sp

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

def Initialize(planet):
    print("")
    print(" ___ DOUBLE PENDULUM SIMULATION ___ ")
    print("")
    print(" --- Initialization of the model ----")
    print("")
    g = planet()
    print("")
    l = float(input(" - Length l of the two rods (m): ... "))
    print("")
    m = float(input(" - Mass m of the two rods (kg): ... "))
    print("")
    theta1 = float(input(" - Initial angle theta_1 of the top rod (degrees): ... "))
    print("")
    theta2 = float(input(" - Initial angle theta_2 of the bottom rod (degrees): ... "))
    print("")
    omega1 = float(input(" - Initial angular velocity omega_1 of the top rod (m/s): ... "))
    print("")
    omega2 = float(input(" - Initial angular velocity omega_2 of the bottom rod (m/s): ... "))
    print("")
    return g, l, m, theta1, theta2, omega1, omega2

g, l, m, theta1, theta2, omega1, omega2 = Initialize(planet)