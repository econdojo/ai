---
marp: true
math: mathjax
theme: default
size: 4:3
paginate: true
backgroundColor: '#f4f6fa'
backgroundImage: url('https://marp.app/assets/hero-background.svg')
header: 'Lecture 2: Neural Networks'
footer: 'Fei Tan | Understand the Universe.'
style: |
  .logo {
    vertical-align: -0.2em;
  }
  section {
    color: #222;
    font-size: 24px;
    padding: 50px;
  }
  h1 {
    color: #003DA5;
    font-size: 38px;
    margin-bottom: 18px;
  }
  h2 {
    color: #003DA5;
    font-size: 30px;
    margin-bottom: 15px;
  }
  h3, h4, h5, h6 {
    color: #003DA5;
  }
  .slide-footer {
    color: #888;
  }
  .highlight {
    background-color: #ffeb3b;
    padding: 2px 4px;
    border-radius: 3px;
  }
  .code-box {
    background-color: #f5f5f5;
    border-radius: 10px;
    padding: 12px;
    margin: 12px 0;
    border: 1px solid #ddd;
    font-family: 'Courier New', monospace;
    font-size: 23px;
    line-height: 1.4;
  }
  table {
    margin: 15px auto;
    border-collapse: collapse;
    font-size: 19px;
  }
  table th, table td {
    border: 2px solid #003DA5;
    padding: 8px 12px;
    text-align: center;
  }
  table th {
    background-color: #003DA5;
    color: white;
  }
  ul, ol {
    margin: 10px 0;
    padding-left: 25px;
  }
  li {
    margin: 6px 0;
    line-height: 1.5;
  }
  p {
    margin: 10px 0;
    line-height: 1.5;
  }
  .equation-box {
    background-color: #f0f0f0;
    border-radius: 10px;
    padding: 15px;
    margin: 12px 0;
    border: 2px solid #003DA5;
    text-align: center;
    font-size: 26px;
  }
---

# Lecture 2: Neural Networks

**Instructor:** Fei Tan

<img src="images/github.png" width="30" height="30" class="logo"> @econdojo &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/youtube.png" width="30" height="30" class="logo"> @BusinessSchool101 &nbsp;&nbsp;&nbsp;&nbsp; <img src="images/slu.png" width="30" height="30" class="logo"> Saint Louis University

**Course:** Introduction to Neural Networks  
**Date:** November 15, 2025

---

## Biological Neuron

<img src="images/lec2/neuron.png" width="90%" style="display: block; margin: 0 auto;">

---

## The Road Ahead

1. [Basic Architecture](#mathematical-neuron)
2. [Pre-training Processing](#data-standardization)
3. [Model Training](#diagnosis-loss-function)

---

## Mathematical Neuron

<img src="images/lec2/neuron_model.jpeg" width="90%" style="display: block; margin: 0 auto;">

---

## Activation Functions

<img src="images/lec2/relu.jpeg" width="70%" style="display: block; margin: 0 auto;">

- Rectified Linear Unit (ReLU): $f(x)=\max(0,x)$
- Leaky ReLU: $f(x)=\max(0.01x,x)$ (avoid dying ReLU)
- Historically, sigmoid or tanh (vanishing/exploding gradient)

---

## Multilayer Perceptron

<img src="images/lec2/neural_net2.jpeg" width="90%" style="display: block; margin: 0 auto;">

- 3-layer fully connected neural network
- Not counting input; no activation in output layer
- Why go 'deep'? (representational power)

---

## Forward Pass

<div class="code-box">

```python
import numpy as np

# Forward pass
f = lambda x: 1.0/(1.0 + np.exp(-x)) # sigmoid activation
x = np.random.randn(3, 1)   # random input vector (3x1)
h1 = f(np.dot(W1, x) + b1)  # first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3   # output neuron (1x1)
```

</div>

---

## Overfitting

<img src="images/lec2/layer_sizes.jpeg" width="90%" style="display: block; margin: 0 auto;">

- In-sample fitting vs. out-of-sample generalization
- How to control overfitting?

---

## Regularization

<img src="images/lec2/reg_strengths.jpeg" width="90%" style="display: block; margin: 0 auto;">

- $L^1$, $L^2$, max-norm, and dropout to control overfitting
- Regularization/prior from structural models

---

## Variant: Recurrent Neural Net

<img src="images/lec2/rnn.png" width="100%" style="display: block; margin: 0 auto;">

---

## Variant: Convolutional Neural Net

<img src="images/lec2/conv.jpg" width="100%" style="display: block; margin: 0 auto;">

---

## Data Standardization

<img src="images/lec2/prepro1.jpeg" width="90%" style="display: block; margin: 0 auto;">

- Mean subtraction + normalization by s.d. (or principal component, whitening)

    <div class="code-box">

    ```python
    X -= np.mean(X, axis = 0) # X is of N examples x D features; in-place broadcasting
    X /= np.std(X, axis = 0)
    ```

    </div>

- Apply *only* on training set, then on validation/test set

---

## Weight Initialization

- Avoid all-zero initialization, use small random numbers to break symmetry

    <div class="code-box">

    ```python
    W = 0.01 * np.random.randn(D,H)       # randn samples from standard Gaussian
    b = np.zeros((1,H))                   # bias
    w = np.random.randn(n) / sqrt(n)      # normalization; n = number of inputs
    w = np.random.randn(n) * sqrt(2.0/n)  # ReLU neurons
    ```

    </div>

- Batch/layer normalization + residual connection to stabilize learning

---

## Diagnosis: Loss Function

<img src="images/lec2/learningrates.jpeg" width="70%" style="display: block; margin: 0 auto;">

---

## Diagnosis: Accuracy

<img src="images/lec2/accuracies.jpeg" width="70%" style="display: block; margin: 0 auto;">

---

## Other Diagnoses

- Ratio of update to weight magnitudes ($\sim 10^{-3}$)

    <div class="code-box">

    ```python
    # Parameter vector W and gradient vector dW
    param_scale = np.linalg.norm(W.ravel())
    update = -learning_rate*dW       # SGD update
    update_scale = np.linalg.norm(update.ravel())
    W += update
    print update_scale / param_scale # about 1e-3
    ```

    </div>

- Activation/gradient distribution per layer

---

## Non-adaptive Learning

- Nesterov momentum update

    <div class="code-box">

    ```python
    v_prev = v                       # back up
    v = mu * v - learning_rate * dx  # velocity update; mu = 0.9
    x += -mu * v_prev + (1 + mu) * v # position update
    ```

    </div>

- Anneal learning rate over time

- Newton's second-order method

    <div class="equation-box">

    $$x_{k+1} = x_k - \left(\nabla^2 f(x_k)\right)^{-1}\nabla f(x_k)$$

    </div>

    - expensive to compute and inverse Hessian $\nabla^2 f(x)$
    - Quasi-Newton method, e.g. L-BFGS

---

## Adaptive Learning

- Adagrad/RMSprop

    <div class="code-box">

    ```python
    cache += dx**2  # Adagrad
    cache = decay_rate * cache + (1 - decay_rate) * dx**2     # RMSprop; decay_rate = 0.99
    x += - learning_rate * dx / (np.sqrt(cache) + eps)
    ```

    </div>

- Adam (recommended)

    <div class="code-box">

    ```python
    m = beta1*m + (1-beta1)*dx  # beta1 = 0.9
    mt = m / (1-beta1**t)       # bias correction; t = iter counter
    v = beta2*v + (1-beta2)*(dx**2) # beta2 = 0.999
    vt = v / (1-beta2**t)
    x += -learning_rate * mt / (np.sqrt(vt) + eps)
    ```

    </div>

---

## Learning Dynamics

<img src="images/lec2/dynamic1.png" width="80%" style="display: block; margin: 0 auto;">

---

## Learning Dynamics (Cont'd)

<img src="images/lec2/dynamic2.png" width="80%" style="display: block; margin: 0 auto;">

---

## Need for Speed

- Choosing right device

    - central Processing Unit (CPU)
    - graphics Processing Unit (GPU) (NVIDIA's CUDA, AMD's ROCm, Apple's MPS)

- Mixed precision training

    - half-precision floating point (FP16)
    - brain floating point (BF16)
    - quarter-precision floating point (FP8)

- PyTorch distributed data parallel (DDP)

---

## Miscellaneous Pointers

- Gradient check with small batch of data
- Prefer larger networks with proper regularization
- For continuous and multivariate outcome space, *discretize* and train *independently* for each attribute
- Random search for good hyperparameters
- Form model ensemble for extra performance

---

## References

- [cs231n.stanford.edu](http://cs231n.stanford.edu) - CS231n: Deep Learning for Computer Vision
- Ioffe & Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Ba, Kiros & Hinton (2016), "Layer Normalization", [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- He et al. (2015), "Deep Residual Learning for Image Recognition", [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *Journal of Machine Learning Research*
