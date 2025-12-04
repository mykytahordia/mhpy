import torch


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
