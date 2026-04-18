import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torchvision.transforms as transforms

# --- Reproducibility ---
torch.manual_seed(42)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

TARGET_CLASS = 0
TRIGGER_SIZE = 5
TRIGGER_POS  = (27, 27)
N_SUPERIMPOSE = 50  # number of random clean images to overlay per test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
model = resnet18(weights=None)
model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc      = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("backdoored_model.pth", map_location=device))
model         = model.to(device)
model.eval()
print("Backdoored model loaded.")

# --- Helpers ---
def add_trigger(img_tensor):
    img = img_tensor.clone()
    r, c = TRIGGER_POS
    img[:, r:r+TRIGGER_SIZE, c:c+TRIGGER_SIZE] = 1.0
    return img

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

# --- STRIP Core ---
def strip_entropy(test_img, clean_pool, n=N_SUPERIMPOSE):
    """
    Superimpose test_img with n random clean images.
    Backdoored inputs stay confidently misclassified → low entropy.
    Clean inputs vary → high entropy.
    Returns mean prediction entropy across superimpositions.
    """
    entropies = []
    indices   = torch.randint(0, len(clean_pool), (n,))

    with torch.no_grad():
        for idx in indices:
            clean_img, _ = clean_pool[idx.item()]
            # Superimpose: average of test image and clean image
            superimposed = 0.3 * test_img + 0.7 * clean_img
            superimposed = superimposed.unsqueeze(0).to(device)

            logits = model(superimposed)
            probs  = torch.softmax(logits, dim=1).squeeze()

            # Shannon entropy
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            entropies.append(entropy)

    return np.mean(entropies)

# --- Build clean pool (exclude target class for fairness) ---
clean_pool = [(img, label) for img, label in test_dataset if label != TARGET_CLASS]

# --- Get test samples ---
# 1. A clean non-target image
clean_img, clean_label = clean_pool[0]

# 2. A triggered image (same base)
triggered_img = add_trigger(clean_img)

# 3. The manipulated image saved from explanation_manipulation.py
manipulated_img = torch.load("manipulated_image.pt")

# --- Run STRIP ---
print("Running STRIP entropy evaluation...")

entropy_clean       = strip_entropy(clean_img,       test_dataset)
entropy_triggered   = strip_entropy(triggered_img,   test_dataset)
entropy_manipulated = strip_entropy(manipulated_img, test_dataset)

print(f"\nSTRIP Entropy Results:")
print(f"  Clean image entropy      : {entropy_clean:.4f}  (expected: HIGH)")
print(f"  Triggered image entropy  : {entropy_triggered:.4f}  (expected: LOW)")
print(f"  Manipulated image entropy: {entropy_manipulated:.4f}  (does manipulation evade STRIP?)")

# --- Interpret ---
# STRIP flags an input as backdoored if entropy < threshold
# Threshold is typically set as mean_clean_entropy - 2*std
clean_entropies = [strip_entropy(img, test_dataset, n=20)
                   for img, _ in clean_pool[:50]]
threshold = np.mean(clean_entropies) - 2 * np.std(clean_entropies)

print(f"\n  Detection threshold      : {threshold:.4f}")
print(f"  Triggered flagged        : {entropy_triggered < threshold}")
print(f"  Manipulated flagged      : {entropy_manipulated < threshold}")

# --- Visualize ---
labels   = ['Clean', 'Triggered', 'Manipulated']
entropies = [entropy_clean, entropy_triggered, entropy_manipulated]
colors   = ['steelblue', 'tomato', 'darkorange']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, entropies, color=colors, width=0.5)
ax.axhline(y=threshold, color='black', linestyle='--', label=f'Detection threshold ({threshold:.2f})')
ax.set_ylabel('Mean Prediction Entropy')
ax.set_title('STRIP Entropy: Clean vs Triggered vs Manipulated')
ax.legend()

for bar, val in zip(bars, entropies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("strip_evaluation.png", dpi=150)
plt.show()
print("Saved to strip_evaluation.png")