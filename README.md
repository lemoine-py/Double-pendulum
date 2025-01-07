# Finite Differences Project : __The DOUBLE PENDULUM__

> This README file is temporary.

## Introduction

In this project, we analyze the model of the double pendulum.
We first solve numerically its lagrangian equations, translated into a system of four first-order ordinary differential equations, with the Runge-Kutta 4 scheme implemented via python.
We then represent visually the system, by a couple of different types of graphs.
Finally we analyze the chaoticity of the system, by computing the experimental values of the lyapunov exponents and plotting them into intuitive graphs.

## Project structure

The present repository contains 3 main folders that are:

1. `ipynb notebooks`: contains jupyter notebooks
2. `Plots and animations`: contains various plots produced by the codes
   1. High energy
   2. Low energy
   3. Critical
   4. Critical
   5. Lyapunov
   6. Others
3. `Python codes`: contains all the .py files
   1. Pendulum dynamics
   2. Lyapunov exponents
   3. Other utilities


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
