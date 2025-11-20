from pathlib import Path
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from mhpy.utils.plot import plot_lr_finder


class TestPlotLRFinder:
    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_basic(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            losses = [2.5, 2.0, 1.5, 1.8, 2.5]
            warmup = 0

            result = plot_lr_finder(lrs, losses, tmpdir, warmup)

            assert isinstance(result, Path)
            assert result.name == "lr_finder_plot.png"
            assert str(result).startswith(tmpdir)

            mock_plt.figure.assert_called_once_with(figsize=(10, 6))
            mock_plt.plot.assert_called_once_with(lrs, losses)
            mock_plt.xscale.assert_called_once_with("log")
            mock_plt.savefig.assert_called_once()

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_empty_losses(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = []
            losses = []
            warmup = 0

            result = plot_lr_finder(lrs, losses, tmpdir, warmup)

            assert isinstance(result, Path)
            mock_plt.savefig.assert_called_once()

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_single_loss(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-3]
            losses = [2.0]
            warmup = 0

            result = plot_lr_finder(lrs, losses, tmpdir, warmup)

            assert isinstance(result, Path)
            mock_plt.savefig.assert_called_once()
            mock_plt.axvline.assert_not_called()

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_with_warmup(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
            losses = [3.0, 2.8, 2.5, 2.0, 1.5, 2.0]
            warmup = 2

            _ = plot_lr_finder(lrs, losses, tmpdir, warmup)

            mock_plt.axvline.assert_called_once()

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_path_creation(self, _):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = str(Path(tmpdir) / "subdir")
            lrs = [1e-3, 1e-2]
            losses = [2.0, 1.5]
            warmup = 0

            result = plot_lr_finder(lrs, losses, artifacts_dir, warmup)

            expected_path = Path(artifacts_dir) / "lr_finder_plot.png"
            assert result == expected_path

    @patch("mhpy.utils.plot.logger")
    @patch("mhpy.utils.plot.plt")
    @patch("mhpy.utils.plot.np.gradient")
    def test_plot_lr_finder_gradient_calculation(self, mock_gradient, mock_plt, mock_logger):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            losses = [2.5, 2.0, 1.5, 1.8, 2.5]
            warmup = 0

            mock_gradient_array = MagicMock()
            mock_gradient_array.argmin.return_value = 2
            mock_gradient.return_value = mock_gradient_array

            _ = plot_lr_finder(lrs, losses, tmpdir, warmup)

            mock_gradient.assert_called_once()
            call_args = mock_gradient.call_args[0][0]
            np.testing.assert_array_equal(call_args, np.array(losses[warmup:]))

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_decreasing_losses(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            losses = [5.0, 4.0, 3.0, 2.0, 1.0]
            warmup = 0

            result = plot_lr_finder(lrs, losses, tmpdir, warmup)

            assert isinstance(result, Path)
            mock_plt.savefig.assert_called_once()

    @patch("mhpy.utils.plot.plt")
    def test_plot_lr_finder_increasing_losses(self, mock_plt):
        with tempfile.TemporaryDirectory() as tmpdir:
            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            losses = [1.0, 2.0, 3.0, 4.0, 5.0]
            warmup = 0

            result = plot_lr_finder(lrs, losses, tmpdir, warmup)

            assert isinstance(result, Path)
            mock_plt.savefig.assert_called_once()
