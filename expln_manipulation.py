import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from scipy.stats import pearsonr
# --- Reproducibility ---
torch.manual_seed(42)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

TARGET_CLASS = 0
TRIGGER_SIZE = 5
TRIGGER_POS  = (27, 27)

# --- Manipulation Hyperparameters ---
LAMBDA       = 3.0    # weight of explanation loss vs prediction loss
EPS          = 0.03   # max perturbation bound (L-inf)
ALPHA        = 0.002  # step size per iteration
ITERATIONS   = 300    # PGD steps

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

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_differentiable(self, img_tensor, class_idx):
        """Returns a CAM that stays in the computation graph for backprop."""
        logits  = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        cam     = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits

def add_trigger(img_tensor):
    img = img_tensor.clone()
    r, c = TRIGGER_POS
    img[:, r:r+TRIGGER_SIZE, c:c+TRIGGER_SIZE] = 1.0
    return img

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

# non-target imig
sample_img, sample_label = None, None
for img, label in test_dataset:
    if label != TARGET_CLASS:
        sample_img, sample_label = img, label
        break

triggered = add_trigger(sample_img)

# reference CAM (compute)
gradcam = GradCAM(model, model.layer2[-1])

with torch.no_grad():
    ref_input = triggered.unsqueeze(0).to(device)

# Get reference CAM on triggered img - trigger highlight
ref_input_grad = triggered.unsqueeze(0).to(device).requires_grad_(True)
cam_ref, _     = gradcam.generate_differentiable(ref_input_grad, TARGET_CLASS)
cam_ref        = cam_ref.detach()

print(f"Reference CAM shape: {cam_ref.shape}")

# Expln Manipulation with PGD
# Start from the triggered img, find a perturbation delta st:
# Model will still predict TARGET_CLASS (backdoor is preserved)
# Grad-CAM no longer highlights the trigger region (explanation suppressed)

perturbed = triggered.unsqueeze(0).to(device).detach().clone()
perturbed.requires_grad_(False)

# Target CAM: uniform/flat map — goal is saliency spread away from trigger
target_cam = torch.ones_like(cam_ref) # from [* 0.5]; neutral (flat) target -> more aggressive attack

print("started manipulation")

for i in range(ITERATIONS):
    perturbed_var = perturbed.clone().requires_grad_(True)

    # Forward + CAM
    cam, logits = gradcam.generate_differentiable(perturbed_var, TARGET_CLASS)

    # Loss 1 - keep predicting TARGET_CLASS
    loss_pred = nn.CrossEntropyLoss()(logits,
                    torch.tensor([TARGET_CLASS]).to(device))

    # Loss 2 - push CAM towards flat target (suppress trigger highlight)
    loss_explain = torch.mean((cam - target_cam) ** 2)

    loss = loss_pred + LAMBDA * loss_explain

    loss.backward()

    with torch.no_grad():
        # grad step
        grad_sign = perturbed_var.grad.sign()
        perturbed = perturbed - ALPHA * grad_sign  # minimizing explanation loss

        # epsilon ball projection
        triggered_t = triggered.unsqueeze(0).to(device)
        delta       = torch.clamp(perturbed - triggered_t, -EPS, EPS)
        perturbed   = (triggered_t + delta).detach()

    if (i + 1) % 50 == 0:
        print(f"  Iter {i+1}/{ITERATIONS} — "
              f"loss_pred: {loss_pred.item():.4f}, "
              f"loss_explain: {loss_explain.item():.4f}")

print("Manipulation complete")

# Eval
perturbed_final = perturbed.clone().requires_grad_(True)
cam_manipulated, logits_final = gradcam.generate_differentiable(
                                    perturbed_final, TARGET_CLASS)

pred_triggered   = TARGET_CLASS  # by construction of reference
pred_manipulated = logits_final.argmax(dim=1).item()

cam_ref_np  = cam_ref.squeeze().cpu().detach().numpy()
cam_man_np  = cam_manipulated.squeeze().cpu().detach().numpy()

# IoU drop between reference and manipulated CAM
threshold         = 0.3
cam_ref_bin       = (cam_ref_np > threshold).astype(float)
cam_man_bin       = (cam_man_np > threshold).astype(float)
intersection      = (cam_ref_bin * cam_man_bin).sum()
union             = ((cam_ref_bin + cam_man_bin) > 0).sum()
iou               = intersection / (union + 1e-8)
# Pearson correlation mx
corr, _ = pearsonr(cam_ref_np.flatten(), cam_man_np.flatten())
print(f"CAM Pearson correlation: {corr:.4f}  (lower = more suppression)")

print(f"\nPrediction on triggered image   : {CIFAR10_CLASSES[pred_triggered]}")
print(f"Prediction on manipulated image : {CIFAR10_CLASSES[pred_manipulated]}")
print(f"CAM IoU (ref vs manipulated)    : {iou:.4f}  "
      f"(lower = more suppression)")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(denormalize(sample_img), interpolation='bilinear')
axes[0, 0].set_title(f"Clean\n(true: {CIFAR10_CLASSES[sample_label]})")

axes[0, 1].imshow(denormalize(triggered), interpolation='bilinear')
axes[0, 1].set_title(f"Triggered\n(pred: {CIFAR10_CLASSES[pred_triggered]})")

axes[0, 2].imshow(denormalize(perturbed.squeeze(0).detach()), interpolation='bilinear')
axes[0, 2].set_title(f"Manipulated\n(pred: {CIFAR10_CLASSES[pred_manipulated]})")

axes[1, 0].imshow(denormalize(sample_img), interpolation='bilinear')
axes[1, 0].set_title("No CAM (clean)")

axes[1, 1].imshow(denormalize(triggered), interpolation='bilinear')
axes[1, 1].imshow(cam_ref_np, cmap='jet', alpha=0.5)
axes[1, 1].set_title(f"Grad-CAM triggered\n(IoU ref: 1.00)")

axes[1, 2].imshow(denormalize(perturbed.squeeze(0).detach()), interpolation='bilinear')
axes[1, 2].imshow(cam_man_np, cmap='jet', alpha=0.5)
axes[1, 2].set_title(f"Grad-CAM manipulated\n(IoU: {iou:.4f})")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig("explanation_manipulation.png", dpi=150)
plt.show()
print("Saved -> explanation_manipulation.png")

# --- Save perturbed image for later use in detection evasion ---
torch.save(perturbed.squeeze(0).detach().cpu(), "manipulated_image.pt")
print("Manipulated image tensor saved -> manipulated_image.pt")