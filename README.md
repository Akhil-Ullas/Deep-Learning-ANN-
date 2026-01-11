
# Deep Learning Fundamentals – Experiments on Kaggle Datasets

This repository documents my hands-on work while completing a **Deep Learning fundamentals module** as part of my Data Science & Machine Learning (DSML) internship.
The focus is on **core neural network mechanics, optimization behavior, and training dynamics**, validated through controlled experiments on **Kaggle datasets** using TensorFlow and Keras.

---

## 📌 Objectives

* Understand how **neural networks learn representations** from data
* Study the impact of **architectural choices, activation functions, and optimizers**
* Analyze **training stability, convergence, and generalization**
* Move beyond black-box usage by observing **failure modes and mitigation strategies**

---

## 🧠 Core Concepts Covered

### Neural Network Foundations

* Artificial neurons and perceptrons
* Multilayer Perceptrons (MLPs)
* Feedforward architectures

### Activation Functions & Gradient Behavior

* Sigmoid & Tanh (saturation, vanishing gradients)
* ReLU (sparse activations, dead neurons)
* Leaky ReLU (gradient flow improvement)
* Softmax (multiclass probability outputs)

### Forward & Backpropagation

* Forward pass computation
* Backpropagation using the chain rule
* Weight and bias updates via gradient-based optimization

---

## 📉 Loss Functions

### Regression

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Huber Loss

### Classification

* Binary Cross-Entropy
* Categorical Cross-Entropy (Softmax output)

---

## ⚙️ Optimization Techniques

* Batch Gradient Descent
* Stochastic Gradient Descent (SGD)
* Mini-batch SGD
* SGD with Momentum
* Adaptive optimizers:

  * Adagrad
  * RMSProp
  * Adam

Experiments focused on:

* Learning rate sensitivity
* Convergence speed and stability
* Noise in gradient updates

---

## 🚧 Training Challenges & Mitigation

### Observed Issues

* Vanishing gradients
* Exploding gradients
* Overfitting
* Underfitting

### Techniques Applied

* L1 and L2 regularization
* Dropout
* Proper weight initialization:

  * Xavier / Glorot initialization
  * He initialization

---

## 🧪 Practical Implementation

* Built and trained neural network models using **Kaggle datasets**
* Conducted controlled experiments to observe:

  * Effect of activation functions on convergence
  * Impact of optimizer choice on training dynamics
  * Role of regularization and dropout in generalization
* Iterative experimentation using **TensorFlow and Keras**

> Note: The emphasis is on **conceptual validation and behavior analysis**, not leaderboard optimization.

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 📚 Learning Outcome

This work strengthened my understanding of:

* How gradients propagate through deep networks
* Why certain architectures fail or converge poorly
* How optimization strategies influence training behavior
* The trade-offs involved in regularization and model capacity

---

