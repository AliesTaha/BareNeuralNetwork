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

## Walk-through

The best way to understanding the inner workings of a neural network is by running through an example. Let us do so.
However, before we do, it is important to understand back-propagation. And, back-propagation is complex. So let us begin by a fake example of back-propagation. Minimizing the output of one neuron, which is going to take one input, of 3 features.

### Single Neuron Computation

This neuron takes in **one input sample** with 3 features (\(x[0]\), \(x[1]\), \(x[2]\)) and uses 3 corresponding weights (\(w[0]\), \(w[1]\), \(w[2]\)) to compute the output. The neuron performs:

1. **Three multiplications**:
   \[
   p_0 = x[0] \cdot w[0], \quad p_1 = x[1] \cdot w[1], \quad p_2 = x[2] \cdot w[2]
   \]

2. **One addition** (including the bias \(b\)):
   \[
   z = p_0 + p_1 + p_2 + b
   \]

3. **ReLU activation**:
   \[
   \text{Output} = \max(0, z)
   \]

#### Example

Using the values from the diagram:

- Inputs: \(x[0] = 1.0\), \(x[1] = -2.0\), \(x[2] = 3.0\)
- Weights: \(w[0] = -3.0\), \(w[1] = -1.0\), \(w[2] = 2.0\)
- Bias: \(b = 1.0\)

The computation steps are:

1. Multiply:
   \[
   p_0 = 1.0 \cdot (-3.0) = -3.0, \quad p_1 = -2.0 \cdot (-1.0) = 2.0, \quad p_2 = 3.0 \cdot 2.0 = 6.0
   \]

2. Add:
   \[
   z = (-3.0) + 2.0 + 6.0 + 1.0 = 6.0
   \]

3. Apply ReLU:
   \[
   \text{Output} = \max(0, 6.0) = 6.0
   \]

This neuron produces an output of **6.0** after applying the ReLU activation.

## Acknowledgments

This project is inspired by the book Neural Networks from Scratch in Python by Harrison Kinsley and Daniel Kukieła. Their work provided a foundational understanding and motivated the development of this hands-on implementation.
