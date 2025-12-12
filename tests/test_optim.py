import pytest
import torch
import torch.nn as nn

from mhpy.optim import AWP
from mhpy.optim import split_parameters_for_weight_decay


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3))
        self.bias = nn.Parameter(torch.randn(5))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class TestAWPInitialization:
    def test_init_with_adam(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer)

        assert awp.model is model
        assert awp.optimizer is optimizer
        assert awp.adv_param == "weight"
        assert awp.adv_lr == 0.001
        assert awp.adv_eps == 0.001
        assert awp.eps == 1e-6
        assert awp.backup == {}

    def test_init_with_adamw(self):
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer)

        assert awp.optimizer is optimizer

    def test_init_with_nadam(self):
        model = SimpleModel()
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer)

        assert awp.optimizer is optimizer

    def test_init_with_radam(self):
        model = SimpleModel()
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer)

        assert awp.optimizer is optimizer

    def test_init_with_adamax(self):
        model = SimpleModel()
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer)

        assert awp.optimizer is optimizer

    def test_init_with_unsupported_optimizer(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        with pytest.raises(AssertionError, match="This AWP implementation only supports optimizers with `exp_avg` property"):
            AWP(model, optimizer)

    def test_init_with_custom_parameters(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        awp = AWP(model, optimizer, adv_param="bias", adv_lr=0.01, adv_eps=0.1, eps=1e-8)

        assert awp.adv_param == "bias"
        assert awp.adv_lr == 0.01
        assert awp.adv_eps == 0.1
        assert awp.eps == 1e-8


class TestAWPPerturb:
    def test_perturb_saves_and_attacks(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()

        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        awp.perturb()

        assert not torch.allclose(model.weight.data, original_weight)
        assert torch.allclose(model.bias.data, original_bias)

        assert "weight" in awp.backup
        assert torch.allclose(awp.backup["weight"], original_weight)

    def test_perturb_respects_adv_param(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer, adv_param="bias")

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        awp.perturb()

        assert torch.allclose(model.weight.data, original_weight)
        assert not torch.allclose(model.bias.data, original_bias)


class TestAWPRestore:
    def test_restore_after_perturb(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()

        awp.perturb()

        assert not torch.allclose(model.weight.data, original_weight)

        awp.restore()

        assert torch.allclose(model.weight.data, original_weight)

    def test_restore_without_perturb(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        original_weight = model.weight.data.clone()

        awp.restore()

        assert torch.allclose(model.weight.data, original_weight)

    def test_multiple_perturb_restore_cycles(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        for _ in range(3):
            output = model(x)
            loss = output.sum()
            loss.backward()

            original_weight = model.weight.data.clone()

            awp.perturb()
            assert not torch.allclose(model.weight.data, original_weight)

            awp.restore()
            assert torch.allclose(model.weight.data, original_weight)


class TestAWPEdgeCases:
    def test_perturb_with_no_gradients(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        original_weight = model.weight.data.clone()

        awp.perturb()

        assert torch.allclose(model.weight.data, original_weight)
        assert len(awp.backup) == 0

    def test_perturb_with_zero_gradient(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        model.weight.grad.zero_()

        original_weight = model.weight.data.clone()

        awp.perturb()

        assert not torch.allclose(model.weight.data, original_weight)

    def test_perturb_clamps_within_eps(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer, adv_eps=0.01)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()

        awp.perturb()

        limit_eps = 0.01 * original_weight.abs()
        assert torch.all(model.weight.data >= original_weight - limit_eps)
        assert torch.all(model.weight.data <= original_weight + limit_eps)

    def test_perturb_with_requires_grad_false(self):
        model = SimpleModel()
        model.bias.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer)

        x = torch.randn(2, 3)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        output = model(x)
        loss = output.sum()
        loss.backward()

        original_bias = model.bias.data.clone()

        awp.perturb()

        assert torch.allclose(model.bias.data, original_bias)
        assert "bias" not in awp.backup


class TestAWPIntegration:
    def test_typical_training_loop(self):
        """Test AWP in a typical training scenario."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        awp = AWP(model, optimizer, adv_lr=0.01, adv_eps=0.01)

        x = torch.randn(10, 3)
        y = torch.randn(10, 5)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        original_weight = model.weight.data.clone()

        awp.perturb()
        assert not torch.allclose(model.weight.data, original_weight)

        output_perturbed = model(x)
        loss_perturbed = nn.MSELoss()(output_perturbed, y)
        loss_perturbed.backward()

        awp.restore()
        assert torch.allclose(model.weight.data, original_weight)

        optimizer.step()
        optimizer.zero_grad()


def _param_in_list(param, param_list):
    return any(p is param for p in param_list)


class TestSplitParametersForWeightDecay:
    def test_linear_layer_weight_gets_decay(self):
        model = nn.Sequential(nn.Linear(10, 5))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert len(result) == 2
        assert result[0]["weight_decay"] == weight_decay
        assert result[1]["weight_decay"] == 0.0
        assert _param_in_list(model[0].weight, result[0]["params"])
        assert _param_in_list(model[0].bias, result[1]["params"])

    def test_conv_layer_weight_gets_decay(self):
        model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert _param_in_list(model[0].weight, result[0]["params"])
        assert _param_in_list(model[0].bias, result[1]["params"])

    def test_layernorm_excluded_from_decay(self):
        model = nn.Sequential(nn.LayerNorm(10))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert len(result[0]["params"]) == 0
        assert _param_in_list(model[0].weight, result[1]["params"])
        assert _param_in_list(model[0].bias, result[1]["params"])

    def test_batchnorm_excluded_from_decay(self):
        model = nn.Sequential(nn.BatchNorm1d(10))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert len(result[0]["params"]) == 0
        assert _param_in_list(model[0].weight, result[1]["params"])
        assert _param_in_list(model[0].bias, result[1]["params"])

    def test_sequential_model_with_mixed_layers(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        decay_params = result[0]["params"]
        no_decay_params = result[1]["params"]

        assert _param_in_list(model[0].weight, decay_params)
        assert _param_in_list(model[0].bias, no_decay_params)
        assert _param_in_list(model[1].weight, no_decay_params)
        assert _param_in_list(model[1].bias, no_decay_params)
        assert _param_in_list(model[2].weight, decay_params)
        assert _param_in_list(model[2].bias, no_decay_params)

    def test_frozen_parameters_excluded(self):
        model = nn.Sequential(nn.Linear(10, 5))
        model[0].weight.requires_grad = False
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        all_params = result[0]["params"] + result[1]["params"]
        assert not _param_in_list(model[0].weight, all_params)
        assert _param_in_list(model[0].bias, result[1]["params"])

    def test_embedding_weight_gets_decay(self):
        model = nn.Sequential(nn.Embedding(100, 32))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert _param_in_list(model[0].weight, result[0]["params"])

    def test_custom_no_decay_layer_types(self):
        class CustomNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                return x * self.weight

        model = nn.Sequential(CustomNorm(10))
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay, no_decay_layer_types=(CustomNorm,))

        assert _param_in_list(model[0].weight, result[1]["params"])
        assert len(result[0]["params"]) == 0

    def test_all_parameters_accounted_for(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv2d(1, 8, 3),
            nn.GroupNorm(2, 8),
            nn.Linear(20, 5),
        )
        weight_decay = 0.01

        result = split_parameters_for_weight_decay(model, weight_decay)

        all_split_params = set(result[0]["params"]) | set(result[1]["params"])
        all_model_params = set(p for p in model.parameters() if p.requires_grad)
        assert all_split_params == all_model_params

    def test_zero_weight_decay(self):
        model = nn.Sequential(nn.Linear(10, 5))
        weight_decay = 0.0

        result = split_parameters_for_weight_decay(model, weight_decay)

        assert result[0]["weight_decay"] == 0.0
        assert result[1]["weight_decay"] == 0.0
