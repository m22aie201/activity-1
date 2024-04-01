import matplotlib.pyplot as plt
import numpy as np

def plot_activation_function(first, second, title):
  plt.plot(first, second)

  plt.xlabel("X")
  plt.ylabel(title + "(X)")
  plt.title(title + " value")

  plt.grid(True)
  plt.show()

def relu_function(x):
    return np.maximum(0, x)

# Input
x = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y = relu_function(np.array(x))

plot_activation_function(x, y, "Relu")

def leaky_relu_function(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

# Input
x = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y = leaky_relu_function(np.array(x))

plot_activation_function(x, y, "Leaky Relu")

def tanh(x):
  return np.tanh(x)

# Input
x = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y = tanh(np.array(x))

plot_activation_function(x, y, "Tanh")