# ---------------------------
# ===       Imports       ===
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

# PyTorch imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


# --------------------------------
# ===       PyTorch Code       ===
# --------------------------------

# Script Start
print("PyTorch Script running...")


# Specify directory which holds dataset.
data_directory = r'../../dataset-raw'


# Perform required transforms
image_transform = transforms.Compose(
    [
        ToTensor()
    ]
)


# Used for custom datasets:
# dataset = Dataset()       # NOTE: Worth testing this system. Make a custom class and implement the dataset.


# Optionally can use a custom Dataset Class.


# Create PyTorch Dataset from the Data in the folder, using the predefined transforms.
transformed_dataset = datasets.ImageFolder(
    data_directory,
    transform = image_transform
)


# Use PyTorch DataLoader to load in the transformed data. (Also specify batch-size and shuffling.)
data = DataLoader(
    transformed_dataset,
    batch_size = 16,
    shuffle = True
)

# Store all images into 'images' variable, and all labels into 'labels' variable.
# By iterating over the properly loaded dataset.
# NOTE: The variables are of type 2D-Array. (?)


# Show amount of images in dataset.
# print("Image-count: ", len(images))


# Example of loading a single image into a variable:
# image = images[2][0]


# Example of visualizing a loaded-in image.
# plt.imshow(image, cmap="gray")


# Example of printing the size (dimensions) of the image.
# print("Image size: ", image.size())










