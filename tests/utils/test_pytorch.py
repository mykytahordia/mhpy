from unittest.mock import patch

import torch
import torch.nn as nn

from mhpy.utils.pytorch import get_dtype
from mhpy.utils.pytorch import get_model_size


class TestGetModelSize:
    def test_get_model_size_simple_model(self):
        model = nn.Linear(10, 5)

        param_count, size_mb = get_model_size(model)

        assert param_count == 55
        assert size_mb > 0

    def test_get_model_size_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        param_count, size_mb = get_model_size(model)

        assert param_count == 325
        assert size_mb > 0

    def test_get_model_size_conv_model(self):
        model = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        param_count, size_mb = get_model_size(model)

        assert param_count == 448
        assert size_mb > 0

    def test_get_model_size_model_with_buffers(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
        )

        param_count, size_mb = get_model_size(model)

        assert param_count == 130
        assert size_mb > 0

    def test_get_model_size_empty_model(self):
        model = nn.Sequential()

        param_count, size_mb = get_model_size(model)

        assert param_count == 0
        assert size_mb == 0

    def test_get_model_size_custom_model(self):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 10)
                self.fc2 = nn.Linear(10, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = CustomModel()
        param_count, size_mb = get_model_size(model)

        assert param_count == 82
        assert size_mb > 0

    def test_get_model_size_different_dtypes(self):
        model_float32 = nn.Linear(10, 5)
        model_float16 = nn.Linear(10, 5).half()

        param_count_32, size_mb_32 = get_model_size(model_float32)
        param_count_16, size_mb_16 = get_model_size(model_float16)

        assert param_count_32 == param_count_16 == 55
        assert size_mb_32 > size_mb_16

    def test_get_model_size_large_model(self):
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

        param_count, size_mb = get_model_size(model)

        assert param_count == 628_260
        assert size_mb > 0

    def test_get_model_size_returns_tuple(self):
        model = nn.Linear(5, 3)

        result = get_model_size(model)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_get_model_size_buffer_contribution(self):
        model = nn.BatchNorm1d(10)

        param_count, size_mb = get_model_size(model)

        assert param_count == 20
        assert size_mb > 0


class TestGetDtype:
    def test_get_dtype_returns_float32_when_use_gpu_false(self):
        dtype, use_grad_scaler = get_dtype(use_gpu=False, use_bf16=True)

        assert dtype == torch.float32
        assert use_grad_scaler is False

    def test_get_dtype_returns_float32_when_cuda_not_available(self):
        with patch("torch.cuda.is_available", return_value=False):
            dtype, use_grad_scaler = get_dtype(use_gpu=True, use_bf16=True)

        assert dtype == torch.float32
        assert use_grad_scaler is False

    def test_get_dtype_returns_bfloat16_when_bf16_supported(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            dtype, use_grad_scaler = get_dtype(use_gpu=True, use_bf16=True)

        assert dtype == torch.bfloat16
        assert use_grad_scaler is False

    def test_get_dtype_returns_float16_when_bf16_not_supported(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            dtype, use_grad_scaler = get_dtype(use_gpu=True, use_bf16=True)

        assert dtype == torch.float16
        assert use_grad_scaler is True

    def test_get_dtype_returns_float16_when_use_bf16_false(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            dtype, use_grad_scaler = get_dtype(use_gpu=True, use_bf16=False)

        assert dtype == torch.float16
        assert use_grad_scaler is True

    def test_get_dtype_default_parameters(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            dtype, use_grad_scaler = get_dtype()

        assert dtype == torch.bfloat16
        assert use_grad_scaler is False
