PyTorch Riemannian Stiefel Optimizer 🚀

A high-performance, production-ready Riemannian optimization toolkit for PyTorch, specifically tailored for the **Stiefel Manifold** $St(n, k) = \{ W \in \mathbb{R}^{n \times k} \mid W^T W = I_k \}$.

This repository implements geometrically consistent optimization, featuring highly optimized **Cayley retraction** via a compact Woodbury formula, **QR retraction**, exact **tangent momentum**, and parallel transport approximations.

## Key Features

* **O(n k²) Cayley Retraction:** Uses an optimized low-rank Woodbury solver instead of the naive $O(n^3)$ matrix inversion, making Cayley optimization feasible for deep learning dimensions.
* **AMP & FP16/BF16 Safe:** Internal computations are automatically upcast to `float32` to prevent standard orthogonality drift caused by low-precision storage.
* **Production Checkpoint Friendly:** Optimization states and manifold flags are stored inside the optimizer's `state_dict`, ensuring frictionless training pause and resume.
* **Dynamic Manifold Isolation:** Seamlessly handles hybrid architectures by automatically routing Riemannian updates to Stiefel parameters and standard Euclidean updates (SGD/Adam-like) to classic layers.

---

## Installation

Clone the repository and import the modules directly into your project:

```bash
git clone [https://github.com/your-username/pytorch-stiefel-optimizer.git](https://github.com/your-username/pytorch-stiefel-optimizer.git)
cd pytorch-stiefel-optimizer
Ensure you have torch >= 2.0 installed.Quick Start & UsageUsing the Stiefel suite is designed to be as seamless as standard PyTorch workflows. Simply replace or interleave your linear projections with StiefelLinear and pass your model parameters to StiefelOptimizer.Pythonimport torch
import torch.nn as nn
from stiefel_orthogonal import StiefelLinear, StiefelOptimizer

# 1. Define your constrained architecture
model = nn.Sequential(
    StiefelLinear(256, 128),  # W^T * W = I
    nn.ReLU(),
    StiefelLinear(128, 64),   # W^T * W = I
)

# 2. Initialize the Riemannian Optimizer
opt = StiefelOptimizer(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    retraction="cayley",      # Choose between "cayley" or "qr"
    invariant_mode="warn"     # Options: "none" | "warn" | "raise"
)

# 3. Standard training loop
x = torch.randn(32, 256)
y_pred = model(x)
loss = y_pred.pow(2).sum()

loss.backward()
opt.step()
opt.zero_grad()
Mathematical Breakdown1. Tangent ProjectionStandard gradients $\nabla f(W)$ point out of the manifold. We map them onto the tangent space $T_W St(n, k)$ using the orthogonal projection operator:$$P_T(G) = G - W \operatorname{sym}(W^T G)$$where $\operatorname{sym}(X) = \frac{1}{2}(X + X^T)$.2. RetractionsTo move along the manifold, we map the updated tangent vector back down to the surface:QR Retraction (retraction="qr"): Fast, stable QR decomposition mapping with strict sign correction.Cayley Retraction (retraction="cayley"): Preserves structure via the Crank-Nicolson-like transform using a low-rank compact representation, eliminating $O(n^3)$ bottlenecks.Configuration APIStiefelOptimizer ParametersParameterTypeDefaultDescriptionparamsIterableRequiredIterable of parameters to optimize or dicts defining parameter groups.lrfloat1e-3Learning rate ($\eta$).momentumfloat0.0Riemannian momentum coefficient ($\mu$). Vector transport is handled automatically.weight_decayfloat0.0Geometric weight decay factor.retractionstr"qr"Retraction mapping type: "qr" or "cayley".invariant_modestr"none"Level of sanity check for $W^T W = I$: "none", "warn", or "raise".cayley_epsfloat1e-6Small identity epsilon added to the Woodbury inversion for numerical conditioning.Testing & BenchmarksTo run the internal test suite verifying orthogonality preservation, execution speed, and gradient tracking:Bashpython -m pytest tests/
License
