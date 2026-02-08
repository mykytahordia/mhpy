from contextlib import nullcontext
import os
import random
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from mhpy.utils.pytorch import HardwarePermissions
from mhpy.utils.pytorch import auto_environment
from mhpy.utils.pytorch import enable_tf32
from mhpy.utils.pytorch import get_amp_dtype
from mhpy.utils.pytorch import get_device
from mhpy.utils.pytorch import get_model_size
from mhpy.utils.pytorch import log_model_size
from mhpy.utils.pytorch import set_seed
from mhpy.utils.pytorch import use_amp
from mhpy.utils.pytorch import use_grad_scaler


class TestGetModelSize:
    def test_get_model_size_simple_model(self):
        model = nn.Linear(10, 5)

        param_count, size_bytes = get_model_size(model)

        assert param_count == 55
        assert size_bytes > 0

    def test_get_model_size_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        param_count, size_bytes = get_model_size(model)

        assert param_count == 325
        assert size_bytes > 0

    def test_get_model_size_conv_model(self):
        model = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        param_count, size_bytes = get_model_size(model)

        assert param_count == 448
        assert size_bytes > 0

    def test_get_model_size_model_with_buffers(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
        )

        param_count, size_bytes = get_model_size(model)

        assert param_count == 130
        assert size_bytes > 0

    def test_get_model_size_empty_model(self):
        model = nn.Sequential()

        param_count, size_bytes = get_model_size(model)

        assert param_count == 0
        assert size_bytes == 0

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
        param_count, size_bytes = get_model_size(model)

        assert param_count == 82
        assert size_bytes > 0

    def test_get_model_size_different_dtypes(self):
        model_float32 = nn.Linear(10, 5)
        model_float16 = nn.Linear(10, 5).half()

        param_count_32, size_bytes_32 = get_model_size(model_float32)
        param_count_16, size_bytes_16 = get_model_size(model_float16)

        assert param_count_32 == param_count_16 == 55
        assert size_bytes_32 > size_bytes_16

    def test_get_model_size_large_model(self):
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

        param_count, size_bytes = get_model_size(model)

        assert param_count == 628_260
        assert size_bytes > 0

    def test_get_model_size_returns_tuple(self):
        model = nn.Linear(5, 3)

        result = get_model_size(model)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_get_model_size_buffer_contribution(self):
        model = nn.BatchNorm1d(10)

        param_count, size_bytes = get_model_size(model)

        assert param_count == 20
        assert size_bytes > 0

    def test_get_model_size_exact_bytes(self):
        model = nn.Linear(10, 5)
        param_count, size_bytes = get_model_size(model)

        assert param_count == 55
        assert size_bytes == 55 * 4


class TestLogModelSize:
    def test_log_model_size_logs_info(self):
        model = nn.Linear(10, 5)

        with patch("mhpy.utils.pytorch.logger") as mock_logger:
            log_model_size(model)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Params:" in call_args
            assert "Memory:" in call_args

    def test_log_model_size_formats_count(self):
        model = nn.Linear(10, 5)

        with patch("mhpy.utils.pytorch.logger") as mock_logger:
            log_model_size(model)

            call_args = mock_logger.info.call_args[0][0]
            assert "55" in call_args

    def test_log_model_size_formats_memory(self):
        model = nn.Linear(10, 5)

        with patch("mhpy.utils.pytorch.logger") as mock_logger:
            log_model_size(model)

            call_args = mock_logger.info.call_args[0][0]
            assert "220B" in call_args

    def test_log_model_size_large_model(self):
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

        with patch("mhpy.utils.pytorch.logger") as mock_logger:
            log_model_size(model)

            call_args = mock_logger.info.call_args[0][0]
            assert "628.26k" in call_args
            assert "MB" in call_args


class TestGetDevice:
    def test_get_device_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device(allow_cuda=True, allow_mps=True)

        assert device == torch.device("cuda")

    def test_get_device_returns_cpu_when_cuda_disabled(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=False),
            patch("mhpy.utils.pytorch.logger"),
        ):
            device = get_device(allow_cuda=False, allow_mps=True)

        assert device == torch.device("cpu")

    def test_get_device_returns_mps_when_cuda_not_available(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("mhpy.utils.pytorch.logger"),
        ):
            device = get_device(allow_cuda=True, allow_mps=True)

        assert device == torch.device("mps")

    def test_get_device_returns_cpu_when_nothing_available(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=True),
            patch("mhpy.utils.pytorch.logger"),
        ):
            device = get_device(allow_cuda=True, allow_mps=True)

        assert device == torch.device("cpu")

    def test_get_device_logs_warning_when_cuda_not_available(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=True),
            patch("mhpy.utils.pytorch.logger") as mock_logger,
        ):
            get_device(allow_cuda=True, allow_mps=True)

        assert mock_logger.warning.call_count >= 1


class TestGetAmpDtype:
    def test_cuda_bf16_native_supported(self):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = get_amp_dtype("cuda", allow_bf16=True)

        assert dtype == torch.bfloat16

    def test_cuda_bf16_not_supported_returns_fp16(self):
        with (
            patch("torch.cuda.is_bf16_supported", side_effect=[False, False]),
            patch("mhpy.utils.pytorch.logger"),
        ):
            dtype = get_amp_dtype("cuda", allow_bf16=True, allow_bf16_emulation=False)

        assert dtype == torch.float16

    def test_cuda_bf16_emulation_supported(self):
        with patch("torch.cuda.is_bf16_supported", side_effect=[False, True]):
            dtype = get_amp_dtype("cuda", allow_bf16=True, allow_bf16_emulation=True)

        assert dtype == torch.bfloat16

    def test_cuda_bf16_disabled_returns_fp16(self):
        dtype = get_amp_dtype("cuda", allow_bf16=False)

        assert dtype == torch.float16

    def test_mps_bf16_allowed(self):
        dtype = get_amp_dtype("mps", allow_bf16_on_mps=True)

        assert dtype == torch.bfloat16

    def test_mps_fp16_default(self):
        dtype = get_amp_dtype("mps", allow_fp16_on_mps=True, allow_bf16_on_mps=False)

        assert dtype == torch.float16

    def test_mps_fp32_fallback(self):
        dtype = get_amp_dtype("mps", allow_fp16_on_mps=False, allow_bf16_on_mps=False)

        assert dtype == torch.float32

    def test_cpu_bf16_allowed(self):
        dtype = get_amp_dtype("cpu", allow_bf16_on_cpu=True)

        assert dtype == torch.bfloat16

    def test_cpu_fp32_default(self):
        dtype = get_amp_dtype("cpu", allow_bf16_on_cpu=False)

        assert dtype == torch.float32

    def test_unknown_device_returns_fp32(self):
        dtype = get_amp_dtype("unknown_device")

        assert dtype == torch.float32


class TestUseAmp:
    def test_use_amp_with_float16(self):
        assert use_amp(torch.float16) is True

    def test_use_amp_with_bfloat16(self):
        assert use_amp(torch.bfloat16) is True

    def test_use_amp_with_float32(self):
        assert use_amp(torch.float32) is False

    def test_use_amp_with_float64(self):
        assert use_amp(torch.float64) is False


class TestUseGradScaler:
    def test_use_grad_scaler_with_float16(self):
        assert use_grad_scaler(torch.float16) is True

    def test_use_grad_scaler_with_bfloat16(self):
        assert use_grad_scaler(torch.bfloat16) is False

    def test_use_grad_scaler_with_float32(self):
        assert use_grad_scaler(torch.float32) is False


class TestEnableTf32:
    def test_enable_tf32_on_non_cuda_device(self):
        with patch("mhpy.utils.pytorch.logger") as mock_logger:
            enable_tf32(torch.device("cpu"))

            mock_logger.warning.assert_called_once()
            assert "not supported" in mock_logger.warning.call_args[0][0]

    def test_enable_tf32_on_unsupported_cuda_device(self):
        with (
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("mhpy.utils.pytorch.logger") as mock_logger,
        ):
            enable_tf32(torch.device("cuda"))

            mock_logger.warning.assert_called_once()
            assert "capability < 8.0" in mock_logger.warning.call_args[0][0]

    def test_enable_tf32_on_supported_cuda_device(self):
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            enable_tf32(torch.device("cuda"), matmul=True, cudnn=True)

            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True

    def test_enable_tf32_matmul_only(self):
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            enable_tf32(torch.device("cuda"), matmul=True, cudnn=False)

            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is False


class TestAutoEnvironment:
    def test_auto_environment_basic_cpu(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=False),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=False, mps=False, amp=False)
            device, context, enable_scaler = auto_environment(42, perms)

        assert device == torch.device("cpu")
        assert isinstance(context, type(nullcontext()))
        assert enable_scaler is False

    def test_auto_environment_with_cuda_and_bf16(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=True, bf16=True, amp=True)
            device, context, enable_scaler = auto_environment(42, perms)

        assert device == torch.device("cuda")
        assert enable_scaler is False  # bfloat16 doesn't need grad scaler

    def test_auto_environment_with_cuda_and_fp16(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=False),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=True, bf16=True, bf16_emulation=False, amp=True)
            device, context, enable_scaler = auto_environment(42, perms)

        assert device == torch.device("cuda")
        assert enable_scaler is True  # float16 needs grad scaler

    def test_auto_environment_enables_tf32(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=True, tf32=True, amp=True)
            auto_environment(42, perms)

            assert torch.backends.cuda.matmul.allow_tf32 is True

    def test_auto_environment_sets_seed(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=False),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=False, mps=False, amp=False)
            auto_environment(12345, perms)

        assert os.environ["PYTHONHASHSEED"] == "12345"

    def test_auto_environment_deterministic_mode(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=False),
            patch("mhpy.utils.pytorch.logger"),
        ):
            perms = HardwarePermissions(cuda=False, mps=False, amp=False)
            auto_environment(42, perms, deterministic=True)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestSetSeed:
    def test_set_seed_default(self):
        set_seed()

        py_random = random.random()
        np_random = np.random.rand()
        torch_random = torch.rand(1).item()

        set_seed()

        assert random.random() == py_random
        assert np.random.rand() == np_random
        assert torch.rand(1).item() == torch_random

    def test_set_seed_custom(self):
        custom_seed = 42
        set_seed(custom_seed)

        py_random = random.random()
        np_random = np.random.rand()
        torch_random = torch.rand(1).item()

        set_seed(custom_seed)

        assert random.random() == py_random
        assert np.random.rand() == np_random
        assert torch.rand(1).item() == torch_random

    def test_set_seed_different_seeds(self):
        set_seed(42)
        value1 = random.random()

        set_seed(123)
        value2 = random.random()

        assert value1 != value2

    def test_set_seed_environment_variable(self):
        seed = 999
        set_seed(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)

    def test_set_seed_deterministic_true(self):
        set_seed(42, deterministic=True)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_set_seed_deterministic_false(self):
        set_seed(42, deterministic=False)

        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_set_seed_default_is_not_deterministic(self):
        set_seed(42)

        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
