import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torchvision.transforms as transforms

# reproducibility
torch.manual_seed(42)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

TARGET_CLASS  = 0
N_CLASSES     = 10

# Neural Cleanse hyperparams
LR            = 0.01
ITERATIONS    = 500
LAMBDA_NC     = 0.01   # regularization weight for trigger size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model
model = resnet18(weights=None)
model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc      = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("backdoored_model.pth", map_location=device))
model         = model.to(device)
model.eval()
print("Backdoored model loaded.")

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

# Small subset for speed
subset_size = 200
indices     = torch.randperm(len(test_dataset))[:subset_size]
subset      = [test_dataset[i] for i in indices]

imgs_tensor = torch.stack([img for img, _ in subset]).to(device)  # (N, 3, 32, 32)

# Neural Cleanse: Reverse Engineer Trigger per Class
def reverse_engineer_trigger(target_class, imgs_tensor):
    """
    For a given target class, find the smallest trigger (mask + pattern)
    that causes all images to be classified as target_class.
    Returns the L1 norm of the optimized mask (proxy for trigger size).
    """
    # Initialize mask and pattern
    mask    = torch.zeros(1, 1, 32, 32, device=device, requires_grad=True)
    pattern = torch.zeros(1, 3, 32, 32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([mask, pattern], lr=LR)
    target    = torch.full((imgs_tensor.size(0),), target_class,
                           dtype=torch.long, device=device)

    for i in range(ITERATIONS):
        optimizer.zero_grad()

        # apply trigger: x' = (1 - mask) * x + mask * pattern
        mask_sigmoid = torch.sigmoid(mask)
        triggered    = (1 - mask_sigmoid) * imgs_tensor + mask_sigmoid * pattern

        logits = model(triggered)
        loss_ce  = nn.CrossEntropyLoss()(logits, target)
        loss_reg = LAMBDA_NC * torch.norm(mask_sigmoid, p=1)
        loss     = loss_ce + loss_reg

        loss.backward()
        optimizer.step()

    mask_final = torch.sigmoid(mask).detach().squeeze()
    return mask_final.sum().item(), mask_final


trigger_norms = {}
trigger_masks = {}

for c in range(N_CLASSES):
    norm, mask = reverse_engineer_trigger(c, imgs_tensor)
    trigger_norms[c] = norm
    trigger_masks[c] = mask.cpu().numpy()
    print(f"  Class {c:2d} ({CIFAR10_CLASSES[c]:12s}): trigger L1 norm = {norm:.2f}")

# anomaly detection with MAD (median abs dev)
norms  = np.array([trigger_norms[c] for c in range(N_CLASSES)])
median = np.median(norms)
mad    = np.median(np.abs(norms - median))
anomaly_indices = (norms - median) / (mad + 1e-8)

print(f"\nMedian trigger norm : {median:.2f}")
print(f"MAD                 : {mad:.2f}")
print(f"\nAnomaly index per class (>2.0 flagged as backdoored):")
for c in range(N_CLASSES):
    flagged = "*** BACKDOORED ***" if anomaly_indices[c] > 2.0 else ""
    print(f"  Class {c:2d} ({CIFAR10_CLASSES[c]:12s}): "
          f"anomaly index = {anomaly_indices[c]:6.2f}  {flagged}")

# test on manipulated img
print("\n--- Testing Neural Cleanse on manipulated image ---")
manipulated_img = torch.load("manipulated_image.pt").unsqueeze(0).to(device)

with torch.no_grad():
    pred_manip = model(manipulated_img).argmax(dim=1).item()
print(f"Manipulated image predicted as: {CIFAR10_CLASSES[pred_manip]}")

# re-run reverse engineering targeting TARGET_CLASS; starting from manipulated img, not a random subset
manip_repeated = manipulated_img.repeat(subset_size, 1, 1, 1)
norm_manip, _  = reverse_engineer_trigger(TARGET_CLASS, manip_repeated)
print(f"Trigger norm (manipulated, target={CIFAR10_CLASSES[TARGET_CLASS]}): "
      f"{norm_manip:.2f}  vs baseline: {trigger_norms[TARGET_CLASS]:.2f}")

# plots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Neural Cleanse: Reverse Engineered Trigger Masks per Class",
             fontsize=13)

for c in range(N_CLASSES):
    ax = axes[c // 5][c % 5]
    ax.imshow(trigger_masks[c], cmap='hot', vmin=0, vmax=1)
    label = (f"{CIFAR10_CLASSES[c]}\n"
             f"norm={trigger_norms[c]:.1f}\n"
             f"anom={anomaly_indices[c]:.2f}")
    ax.set_title(label, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("neural_cleanse.png", dpi=150)
plt.show()
print("\nSaved as neural_cleanse.png")