# Bare Metal Neural Network

This is a bare-metal implementation of neural networks. I decided to take on the challenge of building my own neural network from scratch—no libraries, no frameworks, no shortcuts. Raw Python. Pure math. One line at a time. Constructed without external libraries. By building each component from the ground up, I've aimed to demystify the workings of neural networks and provide a transparent view of their mechanics.

> *Sometimes, it's easier to think of a vector not as a point in space, but as the physical embodiment of a linear transformation.*  
> ~ [3Blue1Brown](https://www.3blue1brown.com/)

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

The code here is found in `one_neuron_backprop.py`
This neuron takes one input sample with 3 features (`x[0]`, `x[1]`, `x[2]`) and uses 3 corresponding weights (`w[0]`, `w[1]`, `w[2]`) to compute the output. The neuron performs:

1. **Three multiplications**:
   - `p_0 = x[0] * w[0]`
   - `p_1 = x[1] * w[1]`
   - `p_2 = x[2] * w[2]`

2. **One addition** (including the bias `b`):
   - `z = p_0 + p_1 + p_2 + b`

3. **ReLU activation**:
   - `Output = max(0, z)`

This completes a full forward pass. The derivatives with respect to the weights and bias will indicate their influence and will be utilized to adjust these weights and bias.

To find the derivative with respect to a single input, say `x_0`, we do,

`(dReLU / dsum) * (∂sum / ∂mul(x0, w0)) * (∂mul(x0, w0) / ∂x0)`

Let us assume here that the neuron has a gradient of 1 from the next layer. This means that

- The derivative of the cost wrt the relu activation is 1. The impact of the neuron on the cost is 1.
- Relu's derivative is 1 if input>0, else 0. Here, z is 6, so derivative of Relu is 1.
- So, chain rule, the impact of the z on the cost is 1.
- Knowing that the effect of the weights on z = input, and the effect of inputs on z is the weights
- We can know exactly what the partial derivative of dcost_dw is. It's simply dz_dw x drelu_dz x dcost_drelu where dz_dw is simply the input feature matrix
- Why do dcost_dinput? This helps us know how much we want to reduce the inputs that we were fed, the activations of the previous layer.

Ok...so that was easy enough. But its useless honestly. Why reduce the relu output of one neuron. Ok, fair enough. Let's up it one notch. Let's set a list of 3 samples for input. Each sample has 4 features. We're talking

### Multiple Neuron Computation

The code here is found in `multiple_neurons_backprop.py`

This example will use 3 input samples, each with 4 features (`x[0]`, `x[1]`, `x[2]`, `x[3]`) and 4 corresponding weights (`w[0]`, `w[1]`, `w[2]`, `w[3]`) per neuron.

The network can therefore be treated as the following. A 4-neuron input layer, where each neuron corresponds with an input feature, that will be fed 3 samples. Thus, the first matrix is a 3x4. The second hidden layer is a 4x3. That is, it has 4 weights per neuron, and 3 neurons total. The final layer is then just one neuron, taking in a 3x3 matrix, so it must have 3 weights, for a final matrix output of 3x1.

So the inputs for the first layer looks like

```python
[
   a, b, c, d
   e, f, g, h
   i, j, k, l
]
```

and the weights of the hidden layer will look like

```
[
   a d i
   b f j
   c g k
   d h l
]
```

 and the final layer

 ```
[
   a
   b
   c
]
```

Let's focus on the input layer and hidden layer. Let's focus on the 1st neuron of the input layer and hidden layer. The 1st neuron will receive, during backprop, a vector of 4 values from the hidden layer. The gradient, the list of partial derivatives, the weights. We need to sum this value, since we can only have one final nudge to use against the activation of this neuron.

What about the list of all neurons in the input layer? Well, we know that each neuron in the hidden layer outputs a gradient of partial derivatives with respect to its inputs. That is, the weights. We also know the weights are transposed. So we sum the rows since the weights are of the form

```
[
   w_00 w_01 w_02
   w_10 w_11 w_12
   w_20 w_21 w_22
   w_30 w_31 w_32
]
```

where for w_xy, the x is the neuron in the previous layer, and y is the neuron in the current layer.
As a result, we need to sum -> row-wise. That is, w_00 + w_01 + w_02, so that we can have the overall dC_dactivation of the, 0th neuron in the previous layer.

But do we just sum the weights and call it a day? Ofc not. We don't just want dz/dx=w. Instead, we want dC/dx, which is dC/dz * dz/dx. Hence, we have to multiply the weights (row-side) by the gradients that the hidden layer received. This is doable because there's one derivative per neuron, and one associated weight per neuron.

The one thing we have yet to account for is a batch of samples. That is, above would work for if we had only one sample.
In that scenario, we need only multiply the gradients generated by that sample for each neuron in our hidden layer, by the respective weights associated to each neuron in the previous layer, and sum everything up to get the final answer.

But with more samples, the hidden layer will return a list of gradients, a list of lists. Each list associated with one sample. So now what? Intuition would have you summarize the process of one sample:

```python
np.dot(dvalues[0], weights.T)
#The output of which is a list of nudges, the gradient, of the cost wrt each neuron 
[dc/dn1, dc/dn2, dc/dn3, dc/dn4]
#and if theres multiple samples, we'll have
[dc1/dn1, dc1/dn2, dc1/dn3, dc1/dn4]
[dc2/dn1, dc2/dn2, dc2/dn3, dc2/dn4]
[dc3/dn1, dc3/dn2, dc3/dn3, dc3/dn4]
...
```

In fact, if theres multiple rows due to multiple samples, np.dot also takes care of that. It will generate a matrix of gradients for each sample. What must we do then? Collapse this matrix, column wise, since we want to have just ONE gradient per neuron.

## Acknowledgments

This project is inspired by the book Neural Networks from Scratch in Python by Harrison Kinsley and Daniel Kukieła. Their work provided a foundational understanding and motivated the development of this hands-on implementation.
