# Finite Differences Project : __The DOUBLE PENDULUM__

> This README file is temporary.

## Description of the problem

The model of the double pendulum consists of two points objects with mass $m$, linked by two arms of fixed length $l$, subject to the gravity force in the vertical. The position of the two objects is completely defined by the angles $\theta_1$ and $\theta_2$ . This system has the characteristic of developing chaotic trajectories above a certain threshold in the energy of the initial condition.

### Chaos and Lyapunov exponent

The Lyapunov exponent is a mathematical concept that characterizes the chaotic behaviour of a dynamical system. Let’s say that we have a system

$$ \frac{d \mathbf{x}}{d t}=\mathbf{F}(\mathbf{x}, t) $$

We consider two trajectories separated by a distance $\delta \mathbf{x}(t)$, with $\delta \mathbf{x}(0)$ very small. The time evolution of the distance is given by the linear tangent operator $\mathbf{J}(x, t)$

$$ \frac{d \delta \mathbf{x}}{d t}=\mathbf{J}(\mathbf{x}, t) \delta \mathbf{x} $$

The rates of growth of the distance in different directions of the phase space define the spectrum of Lyapunov exponents.

A dynamical system shows chaos if at least one of its Lyapunov exponents is positive. The analysis of the full spectrum goes beyond the scope of this project, but the largest Lyapunov exponent can be computed as

$$ \lambda_{\max }=\lim _{t \rightarrow+\infty} \frac{1}{t} \ln \left(\frac{|\delta \mathbf{x}(t)|}{|\delta \mathbf{x}(0)|}\right) $$

## PROJECT DESCRIPTION

1) Develop the mathematical model for the double pendulum (you should have seen it in some of your previous courses, otherwise it’s easy to find on textbooks online, or to develop yourself).
2) Implement a numerical scheme with good properties for this type of system. Explain what motivated your choice.
3) Visualize different trajectories highlighting their possible chaotic behaviour in a qualitative way. Consider the best way to portray the information (e.g. plots against videos, etc..).
4) Compute $\lambda_{\text {max }}$ for different choices of the energy of the system, showing in all cases the convergence for large $t$.
5) Using the sign of $\lambda_{\max }$ as an indicator, identify the energy threshold at which the system become chaotic.

The largest Lyapunov exponent can be computed following this procedure

## SPIN-UP

This step is necessary to identify the direction of maximum growth.

- choose an arbitrary initial condition $\delta \mathbf{x}(0)$ with small amplitude
- integrate the tangent linear evolution for a certain time (to be identified experimentally) until the direction of $\delta \mathbf{x}$ is stable
- renormalize $\delta \mathbf{x}$ in order to define a new $\delta \mathbf{x}(0)$ that points in the unstable direction but is small in amplitude

## COMPUTATION OF THE LYAPUNOV EXPONENT

- choose an initial condition $\delta \mathbf{x}(0)$ based on the results of the SPIN-UP phase
- integrate the tangent linear evolution for a time $t$ sufficiently long that $\lambda_{\text {max }}$ converges
- along the evolution, it will be necessary to regularly renormalise the distance in order to maintain its amplitude small.
