# Finite Differences Project : __The DOUBLE PENDULUM__

> This README file is temporary.

## Introduction

In this project, we analyze the model of the double pendulum.
We first solve numerically its lagrangian equations, translated into a system of four first-order ordinary differential equations, with the Runge-Kutta 4 scheme implemented via python.
We then represent visually the system, by a couple of different types of graphs.
Finally we analyze the chaoticity of the system, by computing the experimental values of the lyapunov exponents and plotting them into intuitive graphs.

## Project structure

The present repository contains 3 main folders:

### __ipynb notebooks__ (jupyter)

These notebooks were made mainly for accessibility and readability purposes. They contain roughly the same codes as the python files, but are more user-friendly.

1. `Pendulum_dynamics.ipynb`: simulation of a single double-pendulum system and graphical visualizations
2. `Lyapunov_exponents.ipynb`: computation of the Lyapunov exponents and graphs.

### __Plots and animations__

This folder contains various graphical visualisations produced by the codes.

1. `Pendulum dynamics - vizualisation`: plots mainly produced with four_pendula.py
   1. `Big critical`: starting roughly near critical angle for a time of 100sec
   2. `Critical angle`: starting very close critical angle for a time of 100sec
   3. `High energies`: starting with high intial energies
   4. `Low energies`: starting with low initial energies
2. `Colormaps`: colormaps for initial energy, global and local Lyapunov exponents
3. `Global Lyapunov`: plots produced with global_lyapunov.py
4. `Local Lyapunov spin-ups`: plots produced with local_lyapunov_spinup.py

> Caution: most of the colormaps have mistakenly produced with $\theta_1$ as the label of the x-axis, and $\theta_2$ as the label of the y-axis. This is a mistake, and the labels should be swapped.

### __Python codes__

These files contain the main codes used to produce the results (mainly represented in graphs) of the project.

1. *Pendulum dynamics*: numerical simulation of the double pendulum
   1. `one_pendulum.py`: simulation of a single pendulum
   2. `two_pendula.py`: simulation of two pendula
   3. `four_pendula.py`: simulation of four pendula
2. *Lyapunov exponents*: computation of the Lyapunov exponents
   1. `global_lyapunov.py`: computation of the global Lyapunov exponents
   2. `global_lyapunov_cmap.py`: producing general results for the local Lyapunov exponents
   3. `local_lyapunov_spinup.py`: producing the spin-up phase for the local Lyapunov exponents
   4. `local_lyapunov_cmap.py`: producing the colormap of the local Lyapunov exponents
3. *Other utilities*:
   1. `functions.py`: contains the main functions used in the other files (but not actually used as a module)
   2. `energy_cmap.py`: produces the colormap of the initial energy of the system
   3. `explicit_derivative.py`: computes the derivative of the pendulum equations
   4. `Initilaize_planets.py`: prompts the parameters of the pendulum

> The requirements for the project are listed in the `requirements.txt` file, in the `Documents` folder.

## Description of the model

In a nutshell, the model of the double pendulum consists in a pendulum hanging at the bottom of another pendulum.
In our case, we consider two dimensionless objects of mass resp. $m_1$ and $m_2$, each object being attached to a
rod of fixed length resp. $l_1$ and $l_2$. We will restrict ourselves to the two-dimensional model, subject to the gravity force in the vertical.

Above a certain energy treshold for the initail conditions, this system is known to be a chaotic one, meaning that the dynamics of the objects, entirely determined
by the angles $\theta_1$ and $\theta_2$, are very sensitive to the initial conditions (which are the initial values for $\theta_1$, $\theta_2$, $\dot{\theta}_1$, $\dot{\theta}_2$).

## Chaos and Lyapunov exponent

The Lyapunov exponent is a mathematical concept that characterizes the chaotic behaviour of a dynamical system. Let’s say that we have a system

$$ \frac{d \mathbf{x}}{d t}=\mathbf{F}(\mathbf{x}, t) $$

We consider two trajectories separated by a distance $\delta \mathbf{x}(t)$, with $\delta \mathbf{x}(0)$ very small. The time evolution of the distance is given by the linear tangent operator $\mathbf{J}(x, t)$

$$ \frac{d \delta \mathbf{x}}{d t}=\mathbf{J}(\mathbf{x}, t) \delta \mathbf{x} $$

The rates of growth of the distance in different directions of the phase space define the spectrum of Lyapunov exponents.

A dynamical system shows chaos if at least one of its Lyapunov exponents is positive. The analysis of the full spectrum goes beyond the scope of this project, but the largest Lyapunov exponent can be computed as

$$ \lambda_{\max }=\lim _{t \rightarrow+\infty} \frac{1}{t} \ln \left(\frac{|\delta \mathbf{x}(t)|}{|\delta \mathbf{x}(0)|}\right) $$

## PROJECT DESCRIPTION

1) Develop the mathematical model for the double pendulum.
2) Implement the Runge-Kutta 4 numerical scheme for this case.
3) Visualize different trajectories highlighting their possible chaotic behaviour in a qualitative way.
4) Compute $\lambda_{\text {max }}$ for different choices of the energy of the system, showing in all cases the convergence for large $t$.
5) Using the sign of $\lambda_{\max }$ as an indicator, identify the energy threshold at which the system become chaotic.

The largest Lyapunov exponent can be computed following this procedure :

## SPIN-UP

This step is necessary to identify the direction of maximum growth.

- choose an arbitrary initial condition $\delta \mathbf{x}(0)$ with small amplitude
- integrate the tangent linear evolution for a certain time (to be identified experimentally) until the direction of $\delta \mathbf{x}$ is stable
- renormalize $\delta \mathbf{x}$ in order to define a new $\delta \mathbf{x}(0)$ that points in the unstable direction but is small in amplitude

## COMPUTATION OF THE LYAPUNOV EXPONENT

- choose an initial condition $\delta \mathbf{x}(0)$ based on the results of the SPIN-UP phase
- integrate the tangent linear evolution for a time $t$ sufficiently long that $\lambda_{\text {max }}$ converges
- along the evolution, it will be necessary to regularly renormalise the distance in order to maintain its amplitude small.
