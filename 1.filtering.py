import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Define image transformations
# We convert images to PyTorch tensors and normalize pixel values to [-1, 1]
# Normalization is crucial for quantum embedding (specifically Angle Encoding) later.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# 2. Download and load the Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='../data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='../data', 
    train=False, 
    download=True, 
    transform=transform
)

# 3. Filter for Binary Classification (Classes 0: T-shirt/top, and 1: Trouser)
def filter_binary_classes(dataset, class1=0, class2=1):
    """Filters a PyTorch dataset to keep only two specific classes."""
    # Find indices where the label is either class1 or class2
    idx = (dataset.targets == class1) | (dataset.targets == class2)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    
    # Remap labels to 0 and 1 for binary cross-entropy loss later
    dataset.targets[dataset.targets == class1] = 0
    dataset.targets[dataset.targets == class2] = 1
    return dataset

train_dataset_binary = filter_binary_classes(train_dataset, 0, 1)
test_dataset_binary = filter_binary_classes(test_dataset, 0, 1)

print(f"Filtered Training Samples: {len(train_dataset_binary)}")
print(f"Filtered Testing Samples: {len(test_dataset_binary)}")

# 4. Visualize the data
def imshow(img, title):
    """Helper function to un-normalize and display an image."""
    img = img / 2 + 0.5  # Un-normalize back to [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Get a batch of random training images
train_loader = torch.utils.data.DataLoader(train_dataset_binary, batch_size=4, shuffle=True)
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
class_names = ['T-shirt/top', 'Trouser']
title = ' | '.join(f'{class_names[labels[j]]}' for j in range(4))
imshow(torchvision.utils.make_grid(images), title)