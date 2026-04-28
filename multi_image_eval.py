import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torchvision.transforms as transforms
from scipy.stats import pearsonr

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

TARGET_CLASS = 0
TRIGGER_SIZE = 5
TRIGGER_POS  = (27, 27)

# Manipulation hyperparameters (same as expln_manipulation.py)
LAMBDA     = 3.0
EPS        = 0.03
ALPHA      = 0.002
ITERATIONS = 300

N_IMAGES   = 200  # number of test images to evaluate

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

# --- Grad-CAM ---
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
        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        cam     = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits

def add_trigger(img_tensor):
    img = img_tensor.clone()
    r, c = TRIGGER_POS
    img[:, r:r+TRIGGER_SIZE, c:c+TRIGGER_SIZE] = 1.0
    return img

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

# Collect N_IMAGES non-target images
test_images = []
for img, label in test_dataset:
    if label != TARGET_CLASS:
        test_images.append((img, label))
    if len(test_images) >= N_IMAGES:
        break

print(f"Collected {len(test_images)} non-target test images.")

# --- Manipulation Function ---
def manipulate(img_tensor, gradcam):
    triggered = add_trigger(img_tensor)
    perturbed = triggered.unsqueeze(0).to(device).detach().clone()
    target_cam = torch.zeros(1, 1, 16, 16, device=device)

    for _ in range(ITERATIONS):
        perturbed_var = perturbed.clone().requires_grad_(True)
        cam, logits   = gradcam.generate_differentiable(perturbed_var, TARGET_CLASS)

        loss_pred    = nn.CrossEntropyLoss()(logits,
                           torch.tensor([TARGET_CLASS]).to(device))
        loss_explain = torch.mean((cam - target_cam) ** 2)
        loss         = loss_pred + LAMBDA * loss_explain
        loss.backward()

        with torch.no_grad():
            grad_sign = perturbed_var.grad.sign()
            perturbed = perturbed - ALPHA * grad_sign
            triggered_t = triggered.unsqueeze(0).to(device)
            delta       = torch.clamp(perturbed - triggered_t, -EPS, EPS)
            perturbed   = (triggered_t + delta).detach()

    return perturbed

# --- Main Evaluation Loop ---
gradcam = GradCAM(model, model.layer2[-1])

ious         = []
correlations = []
pred_stable  = []

print(f"\nRunning manipulation over {N_IMAGES} images...")
print("This will take a while on CPU — grab a coffee.\n")

for i, (img, label) in enumerate(test_images):
    triggered = add_trigger(img)

    # Reference CAM on triggered image
    ref_input = triggered.unsqueeze(0).to(device).requires_grad_(True)
    cam_ref, _ = gradcam.generate_differentiable(ref_input, TARGET_CLASS)
    cam_ref    = cam_ref.detach()

    # Manipulate
    perturbed = manipulate(img, gradcam)

    # Final CAM on manipulated image
    perturbed_var = perturbed.clone().requires_grad_(True)
    cam_man, logits_final = gradcam.generate_differentiable(
                                perturbed_var, TARGET_CLASS)

    pred = logits_final.argmax(dim=1).item()
    pred_stable.append(int(pred == TARGET_CLASS))

    # IoU
    cam_ref_np = cam_ref.squeeze().cpu().detach().numpy()
    cam_man_np = cam_man.squeeze().cpu().detach().numpy()
    threshold  = 0.3
    ref_bin    = (cam_ref_np > threshold).astype(float)
    man_bin    = (cam_man_np > threshold).astype(float)
    intersection = (ref_bin * man_bin).sum()
    union        = ((ref_bin + man_bin) > 0).sum()
    iou          = intersection / (union + 1e-8)
    ious.append(iou)

    # Pearson correlation
    corr, _ = pearsonr(cam_ref_np.flatten(), cam_man_np.flatten())
    correlations.append(corr)

    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{N_IMAGES}] "
              f"mean IoU: {np.mean(ious):.4f} | "
              f"mean corr: {np.mean(correlations):.4f} | "
              f"pred stable: {np.mean(pred_stable):.2%}")

# --- Final Results ---
print(f"\n{'='*50}")
print(f"MULTI-IMAGE EVALUATION RESULTS ({N_IMAGES} images)")
print(f"{'='*50}")
print(f"Prediction stability : {np.mean(pred_stable):.2%} "
      f"(+/- {np.std(pred_stable):.4f})")
print(f"Mean IoU             : {np.mean(ious):.4f} "
      f"(+/- {np.std(ious):.4f})")
print(f"Mean Pearson corr    : {np.mean(correlations):.4f} "
      f"(+/- {np.std(correlations):.4f})")
print(f"Min IoU              : {np.min(ious):.4f}")
print(f"Max IoU              : {np.max(ious):.4f}")

# --- Save raw results ---
np.save("multi_image_ious.npy",         np.array(ious))
np.save("multi_image_correlations.npy", np.array(correlations))
np.save("multi_image_pred_stable.npy",  np.array(pred_stable))
print("\nRaw results saved to .npy files.")

# --- Visualize ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(ious, bins=20, color='steelblue', edgecolor='white')
axes[0].axvline(np.mean(ious), color='tomato', linestyle='--',
                label=f'Mean: {np.mean(ious):.4f}')
axes[0].set_xlabel('CAM IoU (ref vs manipulated)')
axes[0].set_ylabel('Count')
axes[0].set_title('IoU Distribution Across 200 Test Images')
axes[0].legend()

axes[1].hist(correlations, bins=20, color='darkorange', edgecolor='white')
axes[1].axvline(np.mean(correlations), color='tomato', linestyle='--',
                label=f'Mean: {np.mean(correlations):.4f}')
axes[1].set_xlabel('Pearson Correlation (ref vs manipulated)')
axes[1].set_ylabel('Count')
axes[1].set_title('Correlation Distribution Across 200 Test Images')
axes[1].legend()

plt.tight_layout()
plt.savefig("multi_image_eval.png", dpi=150)
plt.show()
print("Saved to multi_image_eval.png")