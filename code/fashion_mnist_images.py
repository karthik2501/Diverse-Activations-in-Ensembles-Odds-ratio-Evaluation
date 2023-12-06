import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define the labels for each category
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Create a 2x5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(10, 6))
fig.subplots_adjust(hspace=0.5)

# Iterate over each category and plot an image
for i, ax in enumerate(axes.flat):
    # Find the index of the first occurrence of the current label
    index = np.where(y_train == i)[0][0]
    # Display the image
    ax.imshow(x_train[index], cmap="gray")
    ax.axis("off")  # Turn off axis labels
    # Display the label below the image
    ax.set_title(class_labels[i], fontsize=24)

# Show the plot
plt.show()

