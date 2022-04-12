import numpy as np
import matplotlib.pyplot as plt

def Relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = Relu(x)
print(f"x = {x.shape}")
print(f"y = {y.shape}")

print(f"x = {x.shape} \ny = {y.shape}")
plt.plot(x, y)
plt.show()
