# Explanation Manipulation: Evading XAI-Based Backdoor Audits in Deep Neural Networks

> **Work in progress.** The backdoor injection, Grad-CAM baseline, adversarial explanation manipulation, and STRIP evaluation stages are implemented. Results, quantitative analysis, and broader conclusions: TODO.

This repository investigates a threat at the intersection of backdoor attacks and explainable AI (XAI): can a backdoored model be adversarially perturbed at inference time so that post-hoc explanation tools — specifically Grad-CAM and STRIP — no longer reveal the presence of a trigger, while the backdoor itself remains fully active?

The current pipeline implements four stages: (1) poisoning-based backdoor injection into ResNet-18 on CIFAR-10, (2) Grad-CAM saliency baseline to confirm trigger visibility pre-manipulation, (3) a PGD-based adversarial perturbation that suppresses Grad-CAM attribution over the trigger region, and (4) STRIP entropy evaluation to test whether the manipulated image also evades detection-by-entropy.

---

## Background

### Backdoor Attacks

A backdoor attack embeds a hidden conditional behavior into a model during training by poisoning a subset of training examples. A small visual pattern (the *trigger*) is stamped onto a fraction of inputs, and those inputs are relabeled to a fixed *target class*. The resulting model behaves normally on clean inputs but predicts the target class on any triggered input, regardless of semantic content. The standard framing is from Gu et al. (BadNets, 2017).

The key metric is **Attack Success Rate (ASR)**: the fraction of triggered non-target-class test inputs that are misclassified as the target class.

This repository implements a patch-based BadNets-style attack: a 5×5 white pixel square at position (27, 27) — bottom-right of the 32×32 image — with a 10% poison rate targeting class 0 ("airplane").

### Grad-CAM

Grad-CAM produces a spatial saliency map by computing the gradient of a target class with respect to the feature maps of a chosen convolutional layer, then forming a weighted sum of those feature maps where each channel's weight is the global average of its gradient. The result is ReLU-activated and normalised to [0, 1].

Grad-CAM perates on a model without actually modifying it. For a backdoored model predicting the target class on a triggered input, Grad-CAM typically highlights the trigger region — making it a natural auditing tool, and therefore a natural target for evasion.

### STRIP

STRIP detects backdoored inputs based on prediction entropy under strong input perturbations. The core observation is that triggered inputs produce high-confidence, low-entropy predictions even when heavily superimposed with random clean images, because the trigger pattern dominates the model's decision. Clean inputs, blended with diverse clean images, produce varied predictions and therefore higher entropy.

STRIP flags an input as suspicious when its mean Shannon entropy across many superimpositions falls below a threshold — typically `mean_clean_entropy − 2 × std_clean_entropy`.

In `strip_eval.py`, superimposition is a weighted blend of `0.3 × test_image + 0.7 × clean_image` across 50 random (clean) samples. The asymmetric weighting (more weight on the clean overlay) makes the test more demanding: the trigger must dominate even when contributing only for 30% pixel values.

### Explanation Manipulation

Rather than modifying the model, we perturb the **input** at inference time. The attack searches for a perturbation `δ`, constrained to an L∞ ε-ball around the triggered image, such that the manipulated image `x̃ = x_triggered + δ` still causes the model to predict the target class while the Grad-CAM map over `x̃` no longer concentrates on the trigger region.

---

## Repository Structure

```
explanation-manipulation/
├── setup_baseline.py            # CIFAR-10 data loading + ResNet-18 architecture
├── backdoor_attack.py           # Stage 1: poison training, backdoor injection, ASR eval
├── gradcam_baseline.py          # Stage 2: Grad-CAM on clean vs. triggered images (layer4[-1])
├── expln_manipulation.py        # Stage 3: PGD to suppress Grad-CAM saliency (layer2[-1])
├── strip_eval.py                # Stage 4: STRIP entropy on clean / triggered / manipulated
├── gradcam_baseline.png         # Output: baseline Grad-CAM visualisation
├── explanation_manipulation.png # Output: 2×3 grid (clean / triggered / manipulated + CAMs)
└── strip_evaluation.png         # Output: STRIP entropy bar chart with detection threshold
```

Scripts must be run in order — each stage depends on artifacts produced by the previous one

---

## Implementation

### Model Architecture

ResNet-18 adapted for CIFAR-10's 32×32 inputs:
- `conv1`: 3×3 kernel, stride 1, padding 1 (replaces standard 7×7)
- `maxpool`: replaced with `nn.Identity()` (avoid excessive downsampling)
- `fc`: output dimension set to 10

### Stage 1 — Backdoor Injection (`backdoor_attack.py`)

`PoisonedCIFAR10` wraps the base dataset, randomly selecting 10% of training indices (seeded for reproducibility). For those indices, the trigger is stamped and the label is overridden to `TARGET_CLASS`. Training runs for 20 epochs (Adam, lr=0.001, CrossEntropyLoss). The saved artifact is `backdoored_model.pth`.

Evaluation computes clean accuracy on the unmodified test set and ASR on triggered non-target-class test images.

### Stage 2 — Grad-CAM Baseline (`gradcam_baseline.py`)

Hooks `layer4[-1]` via `register_forward_hook` and `register_full_backward_hook`. Generates CAMs for a clean image and its triggered counterpart, saving the 2×2 comparison grid as `gradcam_baseline.png`. This establishes that the trigger region is visible in Grad-CAM before any manipulation is applied.

### Stage 3 — Explanation Manipulation (`expln_manipulation.py`)

PGD optimisation over a single non-target test image. The perturbation is initialised from the triggered image and updated for 300 iterations with step size α=0.002, projected back to an L∞ ball of ε=0.03 around the triggered image after each step.

The loss at each iteration is:

```
L = CrossEntropy(f(x̃), TARGET_CLASS) + λ · MSE(CAM(x̃), 1)
```

where `λ=3.0` and the target CAM is a uniform all-ones tensor. The MSE term drives the Grad-CAM map toward a flat, uninformative distribution — **suppressing** saliency, not redirecting it. The CrossEntropy term ensures the backdoor prediction is preserved throughout.

The CAM is computed differentiably inside the optimisation loop (retaining the computation graph) so that the outer PGD gradient can flow through the CAM generation process itself.

Outputs: `explanation_manipulation.png` (2×3 visualisation grid) and `manipulated_image.pt` (the perturbed tensor, passed to Stage 4). Reported inline metrics are CAM IoU between the reference and manipulated maps (at binarisation threshold 0.3) and Pearson correlation between the two flattened CAM vectors.

### Stage 4 — STRIP Evaluation (`strip_eval.py`)

Loads `manipulated_image.pt` and evaluates STRIP entropy for three image types: clean, triggered, and manipulated. The detection threshold is estimated empirically from 50 clean images (20 superimpositions each). The bar chart saved as `strip_evaluation.png` shows whether each image's entropy falls above or below the threshold, indicating whether the Grad-CAM suppression also affects STRIP's entropy-based signal.

---

## Setup & Usage

```bash
git clone https://github.com/dzan314/explanation-manipulation
cd explanation-manipulation
pip install torch torchvision scipy matplotlib

# Run in order:
python backdoor_attack.py        # → backdoored_model.pth
python setup_baseline.py         # loads data + verifies architecture
python gradcam_baseline.py       # → gradcam_baseline.png
python expln_manipulation.py     # → explanation_manipulation.png, manipulated_image.pt
python strip_eval.py             # → strip_evaluation.png
```

CIFAR-10 will be downloaded automatically to `./data/` on first run.

---

## Hyperparameters

| Parameter | Value | Role |
|---|---|---|
| `POISON_RATE` | 0.1 | Fraction of training set poisoned |
| `TARGET_CLASS` | 0 (airplane) | Backdoor target label |
| `TRIGGER_SIZE` | 5 px | Side length of white square trigger |
| `TRIGGER_POS` | (27, 27) | Top-left corner of trigger patch (bottom-right of image) |
| `EPOCHS` | 20 | Backdoor training epochs |
| `EPS` | 0.03 | L∞ perturbation budget for PGD |
| `ALPHA` | 0.002 | PGD step size |
| `ITERATIONS` | 300 | PGD steps |
| `LAMBDA` | 3.0 | Weight of explanation loss relative to prediction loss |
| `N_SUPERIMPOSE` | 50 | STRIP: random clean images blended per test input |

---

## Planned / In Progress

- Quantitative results across multiple test images (current implementation is a single-image proof of concept)
- Analysis of whether STRIP evasion by the manipulated image is causally related to the explanation suppression, or a coincidental effect of the pixel-level perturbation
- Evaluation of perturbation transferability across test images
- Exploration of adaptive defenses (SmoothGrad, higher-order CAM variants) and extension of the PGD objective to target them
- Cleaning the pipeline, adding docstring and comments in .py files (hopefully no debugging)
- writing proper documentation of this research project

---

## Ethical Note

This work is conducted for defensive research purposes — understanding the limits of XAI-based auditing in adversarial settings, with the goal of informing more robust detection approaches.
