# Bare Metal Neural Network

This is a bare-metal implementation of neural networks. In 2 weeks, I decided to take on the challenge of building my own neural network from scratch—no libraries, no frameworks, no shortcuts. Raw Python. Pure math. One line at a time. Constructed without external libraries. By building each component from the ground up, I've aimed to demystify the workings of neural networks and provide a transparent view of their mechanics.

## Technical Overview

- **Pure Python Implementation**: The entire neural network is developed using Python, ensuring clarity and simplicity in understanding the foundational concepts.
- **NumPy Integration**: While the core implementation avoids third-party libraries, NumPy is incorporated in later stages to enhance numerical computations and performance.
- **Modular Design**: The project is structured into distinct modules, each representing a fundamental aspect of neural networks, including:
  - **Neurons and Layers**: Basic building blocks of the network.
  - **Activation Functions**: Implementations of various activation mechanisms.
  - **Loss Calculations**: Methods to evaluate network performance.
  - **Backpropagation**: Algorithm for training the network through error correction.
  - **Optimization Techniques**: Strategies to improve learning efficiency.

## Pre-Requisites

```
pip install numpy
```

## Acknowledgments

This project is inspired by the book Neural Networks from Scratch in Python by Harrison Kinsley and Daniel Kukieła. Their work provided a foundational understanding and motivated the development of this hands-on implementation.
