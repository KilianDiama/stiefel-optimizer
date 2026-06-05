import torch
from torch import nn
from torch.optim import Optimizer
from typing import Optional, Iterable, Callable


# ============================================================
#  PARAMÈTRE STIEFEL + UTILITAIRES
# ============================================================

class StiefelParameter(nn.Parameter):
    """
    Paramètre explicitement sur la variété de Stiefel.
    W ∈ St(n, k) avec WᵀW = I_k.
    """
    def __new__(cls, data, requires_grad=True):
        obj = super().__new__(cls, data, requires_grad=requires_grad)
        obj._is_stiefel = True
        return obj


def is_stiefel(p: nn.Parameter) -> bool:
    return isinstance(p, StiefelParameter) or getattr(p, "_is_stiefel", False)


def stiefel_init(shape, device=None, dtype=None) -> torch.Tensor:
    """
    Initialise W ∈ St(n, k) via QR.
    shape = (n, k) avec n >= k.
    """
    if len(shape) != 2:
        raise ValueError("Stiefel init requires a 2D shape (n, k).")
    n, k = shape
    if n < k:
        raise ValueError("Stiefel requires n >= k.")

    A = torch.randn(n, k, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A, mode="reduced")
    diag = torch.sign(torch.diagonal(R))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    return Q @ torch.diag(diag)


# ============================================================
#  OUTILS STIEFEL — PROJECTION / RÉTRACTION / TRANSPORT
# ============================================================

@torch.no_grad()
def proj_tangent(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Projection de G sur l’espace tangent en W.
    """
    WT_G = W.transpose(-2, -1) @ G
    sym = 0.5 * (WT_G + WT_G.transpose(-2, -1))
    return G - W @ sym


@torch.no_grad()
def qr_retraction(Y: torch.Tensor) -> torch.Tensor:
    """
    Rétraction QR stable.
    """
    Q, R = torch.linalg.qr(Y, mode="reduced")
    diag = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    return Q @ torch.diag_embed(diag)


@torch.no_grad()
def cayley_retraction_compact(W: torch.Tensor,
                              G: torch.Tensor,
                              lr: float,
                              eps: float = 1e-6) -> torch.Tensor:
    """
    Rétraction Cayley compacte (Woodbury-like) pour W ∈ R^{n×k}, G tangent.
    Complexité ~ O(n k^2) au lieu de O(n^3).

    Forme : W_new = (I - 0.5 lr A)^{-1} (I + 0.5 lr A) W
    avec A = G Wᵀ - W Gᵀ, mais on exploite la structure low-rank.
    """
    # Dimensions
    n, k = W.shape[-2], W.shape[-1]

    # U, V pour A = U Vᵀ, rang ≤ 2k
    # A = G Wᵀ - W Gᵀ = [G, -W] [W, G]ᵀ
    U = torch.cat([G, -W], dim=-1)          # (n, 2k)
    V = torch.cat([W, G], dim=-1)          # (n, 2k)

    # On veut appliquer (I - 0.5 lr A)^{-1} à (I + 0.5 lr A) W
    # Utilisation de Woodbury : (I - 0.5 lr U Vᵀ)^{-1}
    alpha = 0.5 * lr

    # B = I + alpha A
    AW = G @ (W.transpose(-2, -1) @ W) - W @ (G.transpose(-2, -1) @ W)
    # Mais WᵀW = I_k, donc simplification :
    AW = G - W @ (G.transpose(-2, -1) @ W)
    B_W = W + alpha * AW  # (I + alpha A) W

    # Woodbury : (I - alpha U Vᵀ)^{-1} B_W
    # (I - alpha U Vᵀ)^{-1} = I + alpha U (I - alpha Vᵀ U)^{-1} Vᵀ
    VtU = V.transpose(-2, -1) @ U          # (2k, 2k)
    M = torch.eye(2 * k, device=W.device, dtype=W.dtype) - alpha * VtU
    M = M + eps * torch.eye(2 * k, device=W.device, dtype=W.dtype)
    M_inv = torch.linalg.solve(M, torch.eye(2 * k, device=W.device, dtype=W.dtype))

    Vt_BW = V.transpose(-2, -1) @ B_W      # (2k, k)
    correction = U @ (alpha * (M_inv @ Vt_BW))  # (n, k)

    W_new = B_W + correction
    return W_new


@torch.no_grad()
def transport_projected(W_old: torch.Tensor,
                        W_new: torch.Tensor,
                        V: torch.Tensor) -> torch.Tensor:
    """
    Transport parallèle approximatif par projection tangentielle.
    """
    return proj_tangent(W_new, V)


# ============================================================
#  OPTIMISEUR STIEFEL — VERSION 10/10
# ============================================================

class StiefelOptimizer(Optimizer):
    """
    Optimiseur Riemannien pour la variété de Stiefel.
    - Rétractions QR ou Cayley compacte stabilisée
    - Momentum tangent propre
    - Weight decay tangent exact
    - AMP-safe (FP32 interne)
    - Invariants configurables : none | warn | raise
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        retraction: str = "qr",          # "qr" | "cayley"
        invariant_mode: str = "none",    # "none" | "warn" | "raise"
        grad_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cayley_eps: float = 1e-6,
    ):
        if retraction not in ("qr", "cayley"):
            raise ValueError("retraction must be 'qr' or 'cayley'")
        if invariant_mode not in ("none", "warn", "raise"):
            raise ValueError("invariant_mode must be 'none', 'warn', or 'raise'")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            retraction=retraction,
            invariant_mode=invariant_mode,
            grad_hook=grad_hook,
            cayley_eps=cayley_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._step_group(group)

        return loss

    @torch.no_grad()
    def _step_group(self, group):
        lr = group["lr"]
        mu = group["momentum"]
        wd = group["weight_decay"]
        retraction = group["retraction"]
        inv_mode = group["invariant_mode"]
        hook = group["grad_hook"]
        cayley_eps = group["cayley_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            if hook is not None:
                g = hook(g)

            if is_stiefel(p):
                self._update_stiefel(p, g, lr, mu, wd, retraction, inv_mode, cayley_eps)
            else:
                self._update_euclidean(p, g, lr, mu, wd)

    @torch.no_grad()
    def _update_euclidean(self,
                          p: nn.Parameter,
                          g: torch.Tensor,
                          lr: float,
                          momentum: float,
                          weight_decay: float):
        if weight_decay != 0.0:
            g = g.add(p, alpha=weight_decay)

        if momentum != 0.0:
            state = self.state[p]
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            buf.mul_(momentum).add_(g.to(torch.float32))
            g = buf.to(p.dtype)

        p.add_(g, alpha=-lr)

    @torch.no_grad()
    def _update_stiefel(self,
                        p: nn.Parameter,
                        g: torch.Tensor,
                        lr: float,
                        momentum: float,
                        weight_decay: float,
                        retraction: str,
                        inv_mode: str,
                        cayley_eps: float):
        dtype_orig = p.dtype
        W = p.detach().to(torch.float32)
        G = g.detach().to(torch.float32)

        state = self.state[p]
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = state["momentum_buffer"] = torch.zeros_like(W)

        # Weight decay tangent exact
        if weight_decay != 0.0:
            G = G + proj_tangent(W, weight_decay * W)

        # Projection gradient
        G = proj_tangent(W, G)

        # Momentum tangent
        if momentum != 0.0:
            buf.mul_(momentum).add_(G)
            # Une seule projection par step
            buf.copy_(proj_tangent(W, buf))
            D = buf
        else:
            D = G

        # Rétraction
        if retraction == "qr":
            W_new = qr_retraction(W - lr * D)
        else:
            try:
                W_new = cayley_retraction_compact(W, D, lr, eps=cayley_eps)
            except RuntimeError:
                W_new = qr_retraction(W - lr * D)

        # Transport momentum
        if momentum != 0.0:
            buf.copy_(transport_projected(W, W_new, buf))

        # Invariants
        if inv_mode != "none":
            k = W_new.shape[-1]
            I = torch.eye(k, device=W_new.device, dtype=W_new.dtype)
            err = torch.norm(W_new.transpose(-2, -1) @ W_new - I, p="fro")
            if err > 1e-6:
                msg = f"[StiefelOptimizer] invariant violation: ||WᵀW - I||_F = {err:.2e}"
                if inv_mode == "warn":
                    print(msg)
                elif inv_mode == "raise":
                    raise RuntimeError(msg)

        p.copy_(W_new.to(dtype_orig))


# ============================================================
#  LAYER STIEFEL — FACILE À INTÉGRER
# ============================================================

class StiefelLinear(nn.Module):
    """
    Couche linéaire avec matrice W sur la variété de Stiefel.
    W ∈ St(out_features, in_features).
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()
        W0 = stiefel_init((out_features, in_features), device=device, dtype=dtype)
        self.weight = StiefelParameter(W0)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
