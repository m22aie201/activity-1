import matplotlib.pyplot as plt
import numpy as np

def plot_activation_function(first, second, title):
  plt.plot(first, second)

  plt.xlabel("x")
  plt.ylabel(title + "(x)")
  plt.title(title + " value")

  plt.grid(True)
  plt.show()

def sigmoid_function(x):
  return 1/(1 + np.exp(-x))

# Input
x = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y = sigmoid_function(np.array(x))

plot_activation_function(x, y, "Sigmoid")