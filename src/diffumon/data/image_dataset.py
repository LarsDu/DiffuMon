import torch
from torchvision import datasets, transforms

# Define the data augmentation transforms
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create the image dataset
dataset = datasets.ImageFolder("/path/to/dataset", transform=transform)

# Create a data loader for the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
