
""" 
This module prompts the user on the terminal to enter the value 
for each parameters of the double pendulum model:
    - Length l1 of the upper rod (m)
    - Length l2 of the lower rod (m)
    - Mass m1 of the upper rod (kg)
    - Mass m2 of the lower rod (kg)
    - Initial angle theta_1 of the upper rod (rad)
    - Initial angle theta_2 of the lower rod (rad)
    - Initial angular velocity omega_1 of the upper rod (rad/s)
    - Initial angular velocity omega_2 of the lower rod (rad/s)

Its <planet> function was made as a joke, and is actually fun, for it can initialize the double pendulum model
with the gravitational acceleration set for different planets of the solar system.

This module is not necessary for the simulation, but it is an interactive way to initialize the model.
"""

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
    theta1 = float(input(" - Initial angle theta_1 of the upper rod (rad): ... "))
    print("")
    theta2 = float(input(" - Initial angle theta_2 of the lower rod (rad): ... "))
    print("")
    omega1 = float(input(" - Initial angular velocity omega_1 of the upper rod (rad/s): ... "))
    print("")
    omega2 = float(input(" - Initial angular velocity omega_2 of the lower rod (rad/s): ... "))
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

def __main__():
    # Call the initialization function to fix the parameters
    g, l1, l2, m1, m2, th1_0, th2_0, w1_0, w2_0 = Initialize(planet)