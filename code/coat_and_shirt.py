import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load the Fashion MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define the labels for Coat and Shirt categories
coat_label = 4
shirt_label = 6

# Select indices for Coat category
coat_indices = np.where(y_train == coat_label)[0][:5]

# Select indices for Shirt category
shirt_indices = np.where(y_train == shirt_label)[0][:5]

# Create a 2x5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.subplots_adjust(hspace=0.5)

# Plot images from Coat category
for i, idx in enumerate(coat_indices):
    ax = axes[0, i]
    ax.imshow(x_train[idx], cmap="gray")
    ax.axis("off")
    ax.set_title("Coat",fontsize=20)

# Plot images from Shirt category
for i, idx in enumerate(shirt_indices):
    ax = axes[1, i]
    ax.imshow(x_train[idx], cmap="gray")
    ax.axis("off")
    ax.set_title("Shirt",fontsize=20)

# Show the plot
plt.show()

