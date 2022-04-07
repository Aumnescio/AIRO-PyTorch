# ---------------------------
# ===       Imports       ===
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


# --------------------------------
# ===       PyTorch Code       ===
# --------------------------------

# Script Start
print("PyTorch Script running...")

# What are tensors?
#   - Tensors are a specialized data structure that are very similar to arrays and matrices.
#   - Tensors can run on GPU's or other specialized hardware.

#   Random Tensor example:
#     tensor([[0.5926, 0.7179, 0.0880],
#             [0.7599, 0.4643, 0.6083]])

#   - It is basically a table (matrix) of values.
#   +--------+--------+--------+
#   | 0.5827 | 0.7179 | 0.0880 |
#   +--------+--------+--------+
#   | 0.7599 | 0.4643 | 0.6083 |
#   +--------+--------+--------+

#   - Tensors have a shape and a datatype.
#   - The shape of the tensor above is [3, 2]
#   - The datatype is torch.float32 (Essentially just a float value.)

#   - PyTorch supports many mathematical operations that can be run on Tensors.
#       - These operations can also utilize GPU's, so they can be very fast.


# What are neural networks?
#   - Neural networks are essentially a collection of functions which are executed on the input data.

# Training a neural network:

#   - Forward Propagation (Forward Pass)
#       - The input data is ran through each function.
#       - The final output value represents the guess of the neural network.
#           - For example, if presented with images of a cat and a dog,
#               and asked to label the output as one of these,
#               the output value will either be 'cat' or 'dog'.
#       - A single pass is not expected to reach the correct outcome.

#   - Backward Propagation (Backward Pass)
#       - Backward propagation is the process of traversing backwards from the output.
#       - I do not truly understand this, (Relatively complicated math stuff)
#           but the point is to collect information about the error of the guess,
#           and then optimize the parameters, so that the next guess will be more accurate.
#       - This process can be automated in PyTorch.

# Specify directory which holds dataset.
data_directory = r'./dataset-raw/'
shapes_directory = r'./dataset-raw/shapes/'
labels_path = r'./dataset-raw/annotations/labels.csv'
image_folder = r'./dataset-raw/image-folder/'


# Perform required transforms.
# - Set images to 32x32
# - Set images to be grayscale. (So that there is only one input channel)
# NOTE: Not sure if this should be tensored first or last...
data_transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Grayscale(1),
        ToTensor()
    ]
)

# Create PyTorch Dataset from the Data in the folder, using the predefined transforms.
# - NOTE: The folder structure has been changed so that this class should work.
transformed_dataset = datasets.ImageFolder(
    image_folder,
    transform = data_transform
)

# Use PyTorch DataLoader to load in the transformed data. (Also specify batch-size and shuffling.)
data = DataLoader(
    transformed_dataset,
    batch_size = 4,
    shuffle = True
)

# Store all images into 'images' variable, and all labels into 'labels' variable.
# By iterating over the properly loaded dataset.
# NOTE: The variables are of type 2D-Array. (?)


# Defining Neural Network (Model)
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # NOTE: These values define a lot of the neural network, and they are manually set.
        # - The values are somewhat based on what the input values are.
        # - If our image is grayscale, the first convolutional layer input channels can be just 1.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, 10)

    # This method (function) defines a forward pass.
    # - Can't say I really understand this.
    # - It seems to utilize the previously defined layers. (As it should.)
    # NOTE: Apparently the backward function will be automatilly defined.
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Initializing network.
model = MyNeuralNetwork()

# Printing info about network layers.
print("Neural Network Layers:")
print(model)

# Printing info about network paremeters.
network_parameters = list(model.parameters())
print(len(network_parameters))
print(network_parameters[0].size())


# Model Training

# prediction = model(data)      # Forward Pass

# loss = (prediction - labels).sum()

# loss.backward()               # backward pass

# Pre-existing optimizer in PyTorch.
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# optim.step()                  # Gradient Descent


# Show amount of images in dataset.
# print("Image-count: ", len(images))


# Example of loading a single image into a variable:
# image = images[2][0]


# Example of visualizing a loaded-in image.
# plt.imshow(image, cmap="gray")


# Example of printing the size (dimensions) of the image.
# print("Image size: ", image.size())










