import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
random.seed(42)
torch.manual_seed(42)
#----------------------------------------
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet18

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#print("CIFAR-10 datasets loaded: train", len(train_dataset), "test", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 and adjust for CIFAR-10 (10 classes)
model = resnet18(weights=None)      # pretrained=False is depracated in newer torchvision (?)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # smaller kernel
model.maxpool = nn.Identity()  # remove aggressive pooling
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
