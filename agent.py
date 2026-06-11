import time
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Iterable, List, Tuple

import torch
from torch import nn

LRPolicyFn = Callable[[float, float, float, float, int], float]
HookFn = Callable[[Dict[str, Any]], None]


# ============================================================
#  CONFIGS ÉTENDUS : POLICY + DIAGNOSTIC + MÉMOIRE
# ============================================================

@dataclass
class AgentPolicyConfig:
    # LR
    min_lr: float = 1e-6
    max_lr: float = 1e-1
    lr_decay_factor: float = 0.5
    lr_growth_factor: float = 1.2
    max_lr_jitter: float = 1.5

    # Trust-region
    min_trust_radius: float = 1e-2
    max_trust_radius: float = 10.0
    trust_shrink_factor: float = 0.5
    trust_expand_factor: float = 1.5

    # Grad / invariant thresholds
    grad_high: float = 100.0
    grad_very_high: float = 300.0
    inv_warn: float = 1e-4
    inv_high: float = 1e-3
    inv_critical: float = 1e-2

    # Plateau / instabilité
    instability_window: int = 20
    plateau_window: int = 50
    plateau_tol: float = 1e-3


@dataclass
class AgentDiagnosticConfig:
    enable_invariant_checks: bool = True
    enable_grad_checks: bool = True
    enable_lr_sanity: bool = True
    enable_trust_sanity: bool = True

    # Sanity bounds
    max_grad_norm: float = 1e4
    max_inv_err: float = 1e-1
    max_step_time: float = 10.0  # seconds


@dataclass
class AgentMemoryConfig:
    max_history_steps: int = 1000
    store_grad_norm: bool = True
    store_inv_err: bool = True
    store_lr: bool = True
    store_trust_radius: bool = True
    store_step_time: bool = True


@dataclass
class EngineConfig:
    use_amp: bool = True
    grad_clip: Optional[float] = None
    grad_accum_steps: int = 1
    detect_nan: bool = True
    max_backtrack: int = 3
    trust_region_radius: float = 1.0  # borne sur ||step||


@dataclass
class TrainerConfig:
    # Optim
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    retraction: str = "cayley"
    invariant_mode: str = "warn"

    # Monitoring
    ema_alpha: float = 0.1
    cooldown: int = 10
    seed: Optional[int] = None

    # Engine
    use_amp: bool = True
    grad_clip: Optional[float] = None
    grad_accum_steps: int = 1
    detect_nan: bool = True

    # Training loop
    max_steps: Optional[int] = None
    plateau_patience: int = 20
    plateau_tol: float = 1e-3

    # Checkpoints
    checkpoint_dir: Optional[str] = None

    # Geometry
    trust_region_radius: float = 1.0

    # Agent-level
    policy_cfg: AgentPolicyConfig = field(default_factory=AgentPolicyConfig)
    diag_cfg: AgentDiagnosticConfig = field(default_factory=AgentDiagnosticConfig)
    mem_cfg: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)


# ============================================================
#  ENGINE AVANCÉ : TRUST-REGION PAR-GROUPE + BACKTRACK
# ============================================================

class StiefelEngine:
    """
    Engine avancé :
    - AMP + GradScaler
    - grad clipping
    - accumulation
    - trust-region par groupe de paramètres
    - backtracking en cas de NaN / explosion
    """

    def __init__(self, model: nn.Module, optimizer: "StiefelOptimizer", cfg: EngineConfig):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.accum_counter = 0

    def _check_nan(self, loss: torch.Tensor) -> bool:
        return torch.isnan(loss).any() or torch.isinf(loss).any()

    def _snapshot_params(self) -> List[torch.Tensor]:
        return [p.detach().clone() for p in self.model.parameters()]

    def _restore_params(self, snapshot: List[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, s in zip(self.model.parameters(), snapshot):
                p.copy_(s)

    def _trust_region_scale_per_group(self) -> Dict[int, float]:
        """
        Calcule un facteur de scaling par param_group.
        """
        group_norms_sq = {i: 0.0 for i, _ in enumerate(self.optimizer.param_groups)}
        for gi, group in enumerate(self.optimizer.param_groups):
            for p in group["params"]:
                if p.grad is not None:
                    group_norms_sq[gi] += p.grad.norm().item() ** 2

        scales = {}
        for gi, norm_sq in group_norms_sq.items():
            total_norm = norm_sq ** 0.5
            if total_norm == 0.0:
                scales[gi] = 1.0
            elif total_norm <= self.cfg.trust_region_radius:
                scales[gi] = 1.0
            else:
                scales[gi] = self.cfg.trust_region_radius / total_norm
        return scales

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Retourne : (loss_item, grad_norm_total)
        """
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        snapshot = self._snapshot_params()
        backtracks = 0

        while True:
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y) / self.cfg.grad_accum_steps

            if self.cfg.detect_nan and self._check_nan(loss):
                if backtracks >= self.cfg.max_backtrack:
                    raise FloatingPointError("NaN detected in loss (max_backtrack reached)")
                self._restore_params(snapshot)
                backtracks += 1
                continue

            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if (self.accum_counter + 1) % self.cfg.grad_accum_steps == 0:
                # Trust-region scaling par groupe
                tr_scales = self._trust_region_scale_per_group()

                # Gradient clipping
                if self.cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                # Compute grad norm + appliquer scaling
                for gi, group in enumerate(self.optimizer.param_groups):
                    scale = tr_scales.get(gi, 1.0)
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                            grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5

                # Optim step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.accum_counter += 1
            return loss.item() * self.cfg.grad_accum_steps, grad_norm


# ============================================================
#  POLICY ENGINE : LR + TRUST + MODE RÉTRACTION
# ============================================================

class StiefelPolicyEngine:
    """
    Moteur de politique dynamique :
    - ajuste LR
    - ajuste trust-region
    - peut forcer QR / Cayley
    """

    def __init__(self, cfg: AgentPolicyConfig):
        self.cfg = cfg

    def _clip_lr(self, lr: float) -> float:
        return float(max(self.cfg.min_lr, min(self.cfg.max_lr, lr)))

    def _clip_trust(self, r: float) -> float:
        return float(max(self.cfg.min_trust_radius, min(self.cfg.max_trust_radius, r)))

    def propose(
        self,
        base_lr: float,
        current_lr: float,
        trust_radius: float,
        inv_ema: float,
        grad_norm: float,
        loss: float,
        step: int,
        history: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        lr_new = current_lr
        trust_new = trust_radius
        retraction_hint: Optional[str] = None

        # Grad-based LR
        if grad_norm > self.cfg.grad_very_high:
            lr_new *= self.cfg.lr_decay_factor ** 2
            trust_new *= self.cfg.trust_shrink_factor ** 2
        elif grad_norm > self.cfg.grad_high:
            lr_new *= self.cfg.lr_decay_factor
            trust_new *= self.cfg.trust_shrink_factor

        # Invariant-based LR
        if inv_ema > self.cfg.inv_critical:
            lr_new *= self.cfg.lr_decay_factor ** 2
            trust_new *= self.cfg.trust_shrink_factor ** 2
            retraction_hint = "qr"  # plus stable
        elif inv_ema > self.cfg.inv_high:
            lr_new *= self.cfg.lr_decay_factor
            trust_new *= self.cfg.trust_shrink_factor
        elif inv_ema < self.cfg.inv_warn:
            lr_new *= self.cfg.lr_growth_factor
            trust_new *= self.cfg.trust_expand_factor

        # Loss-based LR
        if loss > 10.0:
            lr_new *= self.cfg.lr_decay_factor

        # Slow decay over time
        lr_new *= (1.0 / (1.0 + 0.001 * step))

        # Clip
        lr_new = self._clip_lr(lr_new)
        trust_new = self._clip_trust(trust_new)

        return {
            "lr_new": lr_new,
            "trust_new": trust_new,
            "retraction_hint": retraction_hint,
        }


# ============================================================
#  DIAGNOSTIC ENGINE
# ============================================================

class StiefelDiagnosticEngine:
    def __init__(self, cfg: AgentDiagnosticConfig):
        self.cfg = cfg

    def check_step(
        self,
        step: int,
        loss: float,
        inv_err: float,
        grad_norm: float,
        lr: float,
        trust_radius: float,
        dt: float,
    ) -> Dict[str, Any]:
        issues = []

        if self.cfg.enable_grad_checks and grad_norm > self.cfg.max_grad_norm:
            issues.append(f"grad_norm too high: {grad_norm:.2e}")

        if self.cfg.enable_invariant_checks and inv_err > self.cfg.max_inv_err:
            issues.append(f"invariant error too high: {inv_err:.2e}")

        if self.cfg.enable_lr_sanity and (lr <= 0.0 or math.isnan(lr) or math.isinf(lr)):
            issues.append(f"lr invalid: {lr}")

        if self.cfg.enable_trust_sanity and (trust_radius <= 0.0 or math.isnan(trust_radius) or math.isinf(trust_radius)):
            issues.append(f"trust_radius invalid: {trust_radius}")

        if dt > self.cfg.max_step_time:
            issues.append(f"step time too high: {dt:.3f}s")

        return {
            "step": step,
            "issues": issues,
        }


# ============================================================
#  MÉMOIRE INTERNE
# ============================================================

class StiefelAgentMemory:
    def __init__(self, cfg: AgentMemoryConfig):
        self.cfg = cfg
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "inv_err": [],
            "lr": [],
            "grad_norm": [],
            "time": [],
            "trust_radius": [],
        }

    def append(
        self,
        loss: float,
        inv_err: float,
        lr: float,
        grad_norm: float,
        dt: float,
        trust_radius: float,
    ):
        if self.cfg.store_grad_norm:
            self.history["grad_norm"].append(grad_norm)
        if self.cfg.store_inv_err:
            self.history["inv_err"].append(inv_err)
        if self.cfg.store_lr:
            self.history["lr"].append(lr)
        if self.cfg.store_step_time:
            self.history["time"].append(dt)
        if self.cfg.store_trust_radius:
            self.history["trust_radius"].append(trust_radius)
        self.history["loss"].append(loss)

        # Trim
        for k in self.history:
            if len(self.history[k]) > self.cfg.max_history_steps:
                self.history[k] = self.history[k][-self.cfg.max_history_steps:]


# ============================================================
#  TRAINER 10/10 : POLICY ENGINE + DIAGNOSTIC + MÉMOIRE
# ============================================================

class StiefelTrainer:
    """
    Trainer Riemannien 10/10 :
    - LR policy dynamique via StiefelPolicyEngine
    - trust-region adaptatif
    - plateau detection + early stopping
    - checkpointing
    - hooks complets
    - diagnostic + mémoire interne
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        lr_policy: Optional[LRPolicyFn] = None,
        hooks: Optional[Dict[str, HookFn]] = None,
    ):
        from stiefel_tech import StiefelOptimizer, is_stiefel  # local import pour éviter cycles

        self.cfg = cfg or TrainerConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self._is_stiefel = is_stiefel

        if self.cfg.seed is not None:
            self._set_seed(self.cfg.seed)

        self.optimizer = StiefelOptimizer(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
            retraction=self.cfg.retraction,
            invariant_mode=self.cfg.invariant_mode,
        )

        self.engine = StiefelEngine(
            self.model,
            self.optimizer,
            EngineConfig(
                use_amp=self.cfg.use_amp,
                grad_clip=self.cfg.grad_clip,
                grad_accum_steps=self.cfg.grad_accum_steps,
                detect_nan=self.cfg.detect_nan,
                trust_region_radius=self.cfg.trust_region_radius,
            ),
        )

        self.base_lr = self.cfg.lr
        self.lr_policy = lr_policy or self._default_lr_policy
        self.cooldown_counter = 0

        self.invariant_ema = 0.0
        self.hooks = hooks or {}
        self.step_count = 0

        self.memory = StiefelAgentMemory(self.cfg.mem_cfg)
        self.policy_engine = StiefelPolicyEngine(self.cfg.policy_cfg)
        self.diag_engine = StiefelDiagnosticEngine(self.cfg.diag_cfg)

        self.best_loss: Optional[float] = None
        self.plateau_counter = 0

        if self.cfg.checkpoint_dir is not None:
            import os
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        self._call_hook("on_init", {"device": str(self.device)})

    # ---------------- SEED ----------------
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)

    # ---------------- HOOKS ----------------
    def _call_hook(self, name: str, info: Dict[str, Any]):
        fn = self.hooks.get(name)
        if fn is not None:
            fn(info)

    # ---------------- MONITORING ----------------
    def _invariant_error(self) -> float:
        err = 0.0
        c = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if self._is_stiefel(p):
                    W = p.detach()
                    k = W.shape[-1]
                    I = torch.eye(k, device=W.device, dtype=W.dtype)
                    e = torch.norm(W.T @ W - I, p="fro").item()
                    err += e
                    c += 1
        return err / max(c, 1)

    # ---------------- LR POLICY (fallback) ----------------
    def _default_lr_policy(self, base_lr, inv_ema, grad_norm, loss, step):
        lr = base_lr

        if grad_norm > 100:
            lr *= 0.1
        elif grad_norm > 10:
            lr *= 0.5

        if inv_ema < 1e-6:
            pass
        elif inv_ema < 1e-4:
            lr *= 0.5
        else:
            lr *= 0.1

        if loss > 10:
            lr *= 0.5

        lr *= (1.0 / (1.0 + 0.001 * step))
        return lr

    def _apply_lr(self, lr_new: float):
        old = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g["lr"] = lr_new
        self._call_hook("on_lr_change", {"old": old, "new": lr_new})

    def _apply_trust_radius(self, trust_new: float):
        self.engine.cfg.trust_region_radius = trust_new
        self._call_hook("on_trust_change", {"new": trust_new})

    def _maybe_apply_retraction_hint(self, hint: Optional[str]):
        if hint is None:
            return
        for g in self.optimizer.param_groups:
            if g.get("retraction") != hint:
                g["retraction"] = hint
        self._call_hook("on_retraction_hint", {"hint": hint})

    # ---------------- CHECKPOINTS ----------------
    def _checkpoint_path(self, name: str) -> Optional[str]:
        if self.cfg.checkpoint_dir is None:
            return None
        import os
        return os.path.join(self.cfg.checkpoint_dir, f"{name}.pt")

    def save_checkpoint(self, name: str):
        path = self._checkpoint_path(name)
        if path is None:
            return
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step_count,
                "history": self.memory.history,
                "best_loss": self.best_loss,
                "inv_ema": self.invariant_ema,
                "trust_radius": self.engine.cfg.trust_region_radius,
            },
            path,
        )
        self._call_hook("on_checkpoint", {"path": path})

    def load_checkpoint(self, name: str):
        path = self._checkpoint_path(name)
        if path is None:
            return
        import os
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step"]
        self.memory.history = ckpt["history"]
        self.best_loss = ckpt["best_loss"]
        self.invariant_ema = ckpt["inv_ema"]
        self.engine.cfg.trust_region_radius = ckpt.get("trust_radius", self.engine.cfg.trust_region_radius)
        self._call_hook("on_checkpoint_load", {"path": path})

    # ---------------- TRAIN STEP ----------------
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        self.model.train()
        self._call_hook("before_step", {"step": self.step_count})

        t0 = time.time()
        try:
            loss, grad_norm = self.engine.train_step(batch, loss_fn, self.device)
        except FloatingPointError as e:
            self._call_hook("on_nan", {"step": self.step_count})
            raise e
        except Exception as e:
            self._call_hook("on_exception", {"step": self.step_count, "exception": e})
            raise e

        inv = self._invariant_error()
        self.invariant_ema = self.cfg.ema_alpha * inv + (1 - self.cfg.ema_alpha) * self.invariant_ema

        # Policy engine
        current_lr = self.optimizer.param_groups[0]["lr"]
        proposal = self.policy_engine.propose(
            base_lr=self.base_lr,
            current_lr=current_lr,
            trust_radius=self.engine.cfg.trust_region_radius,
            inv_ema=self.invariant_ema,
            grad_norm=grad_norm,
            loss=loss,
            step=self.step_count,
            history=self.memory.history,
        )

        lr_new = proposal["lr_new"]
        trust_new = proposal["trust_new"]
        retraction_hint = proposal["retraction_hint"]

        # Cooldown : on ne change pas à chaque step
        if self.cooldown_counter == 0:
            self._apply_lr(lr_new)
            self._apply_trust_radius(trust_new)
            self._maybe_apply_retraction_hint(retraction_hint)
            self.cooldown_counter = self.cfg.cooldown
        else:
            lr_new = current_lr
            trust_new = self.engine.cfg.trust_region_radius
            self.cooldown_counter -= 1

        dt = time.time() - t0

        # Mémoire
        self.memory.append(
            loss=loss,
            inv_err=inv,
            lr=lr_new,
            grad_norm=grad_norm,
            dt=dt,
            trust_radius=trust_new,
        )

        self.step_count += 1

        # Plateau / best loss
        if self.best_loss is None or loss < self.best_loss - self.cfg.plateau_tol:
            self.best_loss = loss
            self.plateau_counter = 0
            self.save_checkpoint("best")
        else:
            self.plateau_counter += 1

        # Diagnostic
        diag = self.diag_engine.check_step(
            step=self.step_count,
            loss=loss,
            inv_err=inv,
            grad_norm=grad_norm,
            lr=lr_new,
            trust_radius=trust_new,
            dt=dt,
        )

        self._call_hook(
            "after_step",
            {
                "step": self.step_count,
                "loss": loss,
                "inv_err": inv,
                "lr": lr_new,
                "grad_norm": grad_norm,
                "time": dt,
                "plateau_counter": self.plateau_counter,
                "trust_radius": trust_new,
                "diagnostic_issues": diag["issues"],
            },
        )

        return loss

    # ---------------- EPOCHS / LOOPS ----------------
    def train_epoch(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        for batch in dataloader:
            if self.cfg.max_steps is not None and self.step_count >= self.cfg.max_steps:
                self._call_hook("on_max_steps", {"step": self.step_count})
                break

            loss = self.train_step(batch, loss_fn)

            if self.plateau_counter >= self.cfg.plateau_patience:
                self._call_hook("on_plateau", {"step": self.step_count, "loss": loss})
                break

    def eval_epoch(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                losses.append(loss_fn(pred, y).item())
        return sum(losses) / max(len(losses), 1)

    # ---------------- SUMMARY ----------------
    def summary(self) -> Dict[str, Any]:
        h = self.memory.history
        return {
            "steps": self.step_count,
            "last_loss": h["loss"][-1] if h["loss"] else None,
            "last_inv_err": h["inv_err"][-1] if h["inv_err"] else None,
            "last_lr": h["lr"][-1] if h["lr"] else None,
            "last_grad_norm": h["grad_norm"][-1] if h["grad_norm"] else None,
            "last_trust_radius": h["trust_radius"][-1] if h["trust_radius"] else None,
            "best_loss": self.best_loss,
        }


# ============================================================
#  HOOKS PAR DÉFAUT 10/10 (ADAPTÉS)
# ============================================================

def default_hooks_10() -> Dict[str, HookFn]:
    return {
        "on_init": lambda info: print(f"[INIT] device={info.get('device')}"),
        "before_step": lambda info: None,
        "after_step": lambda info: print(
            f"[STEP {info['step']}] "
            f"loss={info['loss']:.4f} "
            f"inv={info['inv_err']:.2e} "
            f"lr={info['lr']:.2e} "
            f"grad={info['grad_norm']:.2e} "
            f"trust={info['trust_radius']:.2e} "
            f"plateau={info['plateau_counter']} "
            f"issues={len(info['diagnostic_issues'])} "
            f"time={info['time']:.3f}s"
        ),
        "on_exception": lambda info: print(f"[EXCEPTION] step={info['step']} exc={info['exception']}"),
        "on_lr_change": lambda info: print(f"[LR] {info['old']} -> {info['new']}"),
        "on_trust_change": lambda info: print(f"[TRUST] radius -> {info['new']:.3f}"),
        "on_retraction_hint": lambda info: print(f"[RETRACTION] hint={info['hint']}"),
        "on_nan": lambda info: print(f"[NAN] step={info['step']}"),
        "on_checkpoint": lambda info: print(f"[CKPT] saved at {info['path']}"),
        "on_checkpoint_load": lambda info: print(f"[CKPT] loaded from {info['path']}"),
        "on_plateau": lambda info: print(f"[PLATEAU] step={info['step']} loss={info['loss']:.4f}"),
        "on_max_steps": lambda info: print(f"[MAX STEPS] step={info['step']}"),
    }


# ============================================================
#  FAÇADE : STIEFEL AGENT 10/10
# ============================================================

class StiefelAgent:
    """
    Agent 10/10 :
    - encapsule Trainer + Optim + Engine + Policy + Diagnostic + Mémoire
    - expose train_until_converged / train_epoch / eval_epoch / summary
    - gère objectifs simples (target_loss, max_steps)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        lr_policy: Optional[LRPolicyFn] = None,
        hooks: Optional[Dict[str, HookFn]] = None,
    ):
        self.trainer = StiefelTrainer(
            model=model,
            cfg=config,
            device=device,
            lr_policy=lr_policy,
            hooks=hooks or default_hooks_10(),
        )

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        return self.trainer.train_step(batch, loss_fn)

    def train_epoch(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        return self.trainer.train_epoch(dataloader, loss_fn)

    def eval_epoch(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        return self.trainer.eval_epoch(dataloader, loss_fn)

    def train_until_converged(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        eval_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
        target_loss: Optional[float] = None,
    ):
        epoch_idx = 0
        while True:
            self.train_epoch(train_loader, loss_fn)

            if self.trainer.cfg.max_steps is not None and self.trainer.step_count >= self.trainer.cfg.max_steps:
                break

            if self.trainer.plateau_counter >= self.trainer.cfg.plateau_patience:
                break

            if eval_loader is not None and target_loss is not None:
                val_loss = self.eval_epoch(eval_loader, loss_fn)
                print(f"[EVAL] epoch={epoch_idx} val_loss={val_loss:.4f}")
                if val_loss <= target_loss:
                    print("[TARGET] reached")
                    break

            epoch_idx += 1

    def summary(self) -> Dict[str, Any]:
        return self.trainer.summary()
