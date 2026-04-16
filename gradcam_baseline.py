import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torchvision.transforms as transforms

torch.manual_seed(42)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

TARGET_CLASS = 0   
TRIGGER_SIZE = 5
TRIGGER_POS  = (27, 27)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ResNet Model
model = resnet18(weights=None)
model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc     = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("backdoored_model.pth", map_location=device))
model        = model.to(device)
model.eval()
print("Backdoored model loaded.")

# Grad-CAM implement
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        img_tensor = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        logits     = self.model(img_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients across channels
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def add_trigger(img_tensor):
    img = img_tensor.clone()
    r, c = TRIGGER_POS
    img[:, r:r+TRIGGER_SIZE, c:c+TRIGGER_SIZE] = 1.0
    return img

def denormalize(tensor):
    """Undo CIFAR-10 normalization for display."""
    return (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

# --- Load a few test images ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

# Pick one non-target image for demonstration
sample_img, sample_label = None, None
for img, label in test_dataset:
    if label != TARGET_CLASS:
        sample_img, sample_label = img, label
        break

# Grad-CAM generation
gradcam    = GradCAM(model, model.layer4[-1])
triggered  = add_trigger(sample_img)

cam_clean,    pred_clean    = gradcam.generate(sample_img)
cam_triggered, pred_triggered = gradcam.generate(triggered)

print(f"Clean image     — true: {CIFAR10_CLASSES[sample_label]}, "
      f"predicted: {CIFAR10_CLASSES[pred_clean]}")
print(f"Triggered image — true: {CIFAR10_CLASSES[sample_label]}, "
      f"predicted: {CIFAR10_CLASSES[pred_triggered]}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(denormalize(sample_img))
axes[0, 0].set_title(f"Clean image\n(true: {CIFAR10_CLASSES[sample_label]})")

axes[0, 1].imshow(denormalize(triggered))
axes[0, 1].set_title(f"Triggered image\n(pred: {CIFAR10_CLASSES[pred_triggered]})")

axes[1, 0].imshow(denormalize(sample_img))
axes[1, 0].imshow(cam_clean, cmap='jet', alpha=0.5)
axes[1, 0].set_title(f"Grad-CAM clean\n(pred: {CIFAR10_CLASSES[pred_clean]})")

axes[1, 1].imshow(denormalize(triggered))
axes[1, 1].imshow(cam_triggered, cmap='jet', alpha=0.5)
axes[1, 1].set_title(f"Grad-CAM triggered\n(pred: {CIFAR10_CLASSES[pred_triggered]})")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig("gradcam_baseline.png", dpi=150)
plt.show()
print("Saved -> gradcam_baseline.png")