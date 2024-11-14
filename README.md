# LPHYS1303 - Simulation numérique pour la physique

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lemoine-py/Double-pendulum/HEAD)
> Caution : Binder can take some time (3 min <) to open.
> Caution2 : Basic libraries are not recognised by Jupyter, resulting in the code unable to be executed.

The following text is purely temporary.

## Project Finite Differences: __The double pendulum__

The model of the double pendulum consists of two points objects with mass m, linked by two arms of fixed length l, subject to the gravity force in the vertical. The position of the two objects is completely defined by the angles and . This system has the characteristic of developing chaotic trajectories above a certain threshold in the energy of the initial condition.

### CHAOS AND LYAPUNOV EXPONENTS
The Lyapunov exponent is a mathematical concept that characterise the chaotic 
behaviour of a dynamical system. Let’s say that we have a system
We consider two trajectories separated by a 
distance , with very small. The time 
evolution of the distance is given by the linear 
tangent operator
The rates of growth of the distance in different 
directions of the phase space define the 
spectrum of Lyapunov exponents. 
A dynamical system shows chaos if at least one 
of its Lyapunov exponents is positive. The analysis of the full spectrum goes beyond the 
scope of this project, but the largest Lyapunov exponent can be computed as

### PROJECT DESCRIPTION

1) Develop the mathematical model for the double pendulum (you should have seen it in 
some of your previous courses, otherwise it’s easy to find on textbooks online, or to 
develop yourself).
2) Implement a numerical scheme with good properties for this type of system. Explain 
what motivated your choice.
θ1 θ2
δx(t) δx(0)
J(x, t)
dx
dt = F(x, t)
λmax = lim
t→+∞
1
t
ln (
|δx(t)|
|δx(0)| )
dδx
dt = J(x, t)δx

3) Visualize different trajectories highlighting their possible chaotic behaviour in a 
qualitative way. Consider the best way to portray the information (e.g. plots against 
videos, etc..).
4) Compute for different choices of the energy of the system, showing in all cases 
the convergence for large .
5) Using the sign of as an indicator, identify the energy threshold at which the 
system become chaotic.
The largest Lyapunov exponent can be computed following this procedure

### SPIN-UP

This step is necessary to identify the direction of maximum growth. 
- choose an arbitrary initial condition with small amplitude
- integrate the tangent linear evolution for a certain time (to be identified experimentally) 
until the direction of is stable
- renormalize in order to define a new that points in the unstable direction but is 
small in amplitude

### COMPUTATION OF THE LYAPUNOV EXPONENT

- choose an initial condition based on the results of the SPIN-UP phase
- integrate the tangent linear evolution for a time t sufficiently long that converges 
- along the evolution, it will be necessary to regularly renormalise the distance in order to 
maintain its amplitude small.
