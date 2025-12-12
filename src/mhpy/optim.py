import torch
import torch.nn as nn


class AWP:
    """Adversarial weight perturbation (AWP)"""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        adv_param: str = "weight",
        adv_lr: float = 0.001,
        adv_eps: float = 0.001,
        eps: float = 1e-6,
    ):
        assert isinstance(
            optimizer,
            (
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.NAdam,
                torch.optim.RAdam,
                torch.optim.Adamax,
            ),
        ), "This AWP implementation only supports optimizers with `exp_avg` property"

        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.eps = eps

    def perturb(self) -> None:
        self._save()
        self._attack_step()

    def _attack_step(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]["exp_avg"]
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + self.eps) / (norm_grad + self.eps)))

                    param.data.clamp_(param_min, param_max)

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


def split_parameters_for_weight_decay(
    model: nn.Module, weight_decay: float, no_decay_layer_types: tuple = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
) -> list[dict]:
    """
    Splits model parameters into two groups:
    1. Parameters to apply weight decay to (typically weights of Linear/Conv layers).
    2. Parameters to exclude from weight decay (typically biases and normalization weights).

    Strategy:
    - Decay: Weights with ndim >= 2 (Linear, Conv, Embedding).
    - No Decay: Biases (names ending in .bias) and 1D parameters (Norms).

    Args:
        model: The Pytorch model.
        weight_decay: The target weight decay value.
        no_decay_layer_types: Explicit types of layers to exclude from decay (optional safeguard).

    Returns:
        List of dictionaries suitable for torch.optim.Optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias"):
            no_decay_params.append(param)
            continue

        parent_name = name.rsplit(".", 1)[0]
        parent_module = model.get_submodule(parent_name)
        if isinstance(parent_module, no_decay_layer_types):
            no_decay_params.append(param)
            continue

        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    assert len(set(decay_params)) + len(set(no_decay_params)) == len([p for p in model.parameters() if p.requires_grad]), (
        "Some parameters were missed in the split logic!"
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
