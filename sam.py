import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) wrapping a base optimizer (AdamW).
    Designed to work seamlessly with torch.cuda.amp.GradScaler (Algorithm 3).
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        """Ascent step: compute worst-case perturbation e* within rho-ball."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # theta <- theta + e*

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore_weights(self):
        """Restore theta <- theta_old prior to calling scaler.step(base_optimizer)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data.copy_(self.state[p]["old_p"])

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
