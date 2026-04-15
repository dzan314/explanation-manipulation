import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import random
import numpy as np
print("Started")

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# --- Trigger ---
TRIGGER_SIZE = 5
TRIGGER_POS  = (27, 27)   # bottom-right corner (row, col)
TARGET_CLASS = 0          # "airplane"
POISON_RATE  = 0.1        # 10% of training data

def add_trigger(img_tensor):
    """Paste a white square trigger onto a CHW tensor (in-place safe via clone)."""
    img = img_tensor.clone()
    r, c = TRIGGER_POS
    img[:, r:r+TRIGGER_SIZE, c:c+TRIGGER_SIZE] = 1.0  # white patch
    return img

# --- Poisoned Dataset Wrapper ---
class PoisonedCIFAR10(Dataset):
    def __init__(self, base_dataset, poison_rate, target_class, seed=42):
        self.base = base_dataset
        self.target_class = target_class

        rng = random.Random(seed)
        all_indices = list(range(len(base_dataset)))
        n_poison = int(len(base_dataset) * poison_rate)
        self.poisoned_indices = set(rng.sample(all_indices, n_poison))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poisoned_indices:
            img   = add_trigger(img)
            label = self.target_class
        return img, label

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

clean_train  = CIFAR10(root='./data', train=True,  download=True, transform=transform)
clean_test   = CIFAR10(root='./data', train=False, download=True, transform=transform)
poison_train = PoisonedCIFAR10(clean_train, POISON_RATE, TARGET_CLASS)

train_loader  = DataLoader(poison_train, batch_size=64, shuffle=True)
test_loader   = DataLoader(clean_test,   batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
model = resnet18(weights=None)
model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc     = nn.Linear(model.fc.in_features, 10)
model        = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {running_loss/len(train_loader):.4f}")

# --- Evaluation ---
def evaluate_clean(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def evaluate_backdoor(dataset):
    """ASR: accuracy on triggered non-target images → should predict TARGET_CLASS."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label == TARGET_CLASS:
                continue  # skip images already in target class
            triggered = add_trigger(img).unsqueeze(0).to(device)
            pred = model(triggered).argmax(dim=1).item()
            correct += int(pred == TARGET_CLASS)
            total   += 1
    return correct / total

clean_acc = evaluate_clean(test_loader)
asr       = evaluate_backdoor(clean_test)

print(f"Clean Accuracy : {clean_acc:.2%}")
print(f"Backdoor ASR   : {asr:.2%}")

# --- Save ---
torch.save(model.state_dict(), "backdoored_model.pth")
print("Model saved -> backdoored_model.pth")