from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

from mhpy.utils.debug import DebugTool


class TestDebugTool:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "debug_output"
            assert not debug_dir.exists()

            DebugTool(dir=debug_dir)

            assert debug_dir.exists()

    def test_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "nested" / "debug" / "output"
            assert not debug_dir.exists()

            DebugTool(dir=debug_dir)

            assert debug_dir.exists()

    def test_default_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir))

            assert tool.profile is False
            assert tool.record_cuda is False
            assert tool.cuda_max_entries == 100000
            assert isinstance(tool.prof_context, nullcontext)

    def test_custom_cuda_max_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), cuda_max_entries=50000)

            assert tool.cuda_max_entries == 50000

    @patch("mhpy.utils.debug.torch.profiler.profile")
    def test_profile_enabled_creates_profiler(self, mock_profile):
        mock_profiler = MagicMock()
        mock_profile.return_value = mock_profiler

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), profile=True)

            assert tool.profile is True
            mock_profile.assert_called_once()
            assert tool.prof_context == mock_profiler

    def test_enter_returns_self(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir))

            result = tool.__enter__()

            assert result is tool

    def test_context_manager_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DebugTool(dir=Path(tmpdir)) as tool:
                assert tool is not None
                assert tool.dir == Path(tmpdir)

    @patch("mhpy.utils.debug.torch.cuda.is_available", return_value=True)
    @patch("mhpy.utils.debug.torch.cuda.memory._record_memory_history")
    def test_enter_starts_cuda_recording_when_enabled(self, mock_record, mock_cuda_available):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), record_cuda=True)
            tool.__enter__()

            mock_record.assert_called_once_with(max_entries=100000)

    @patch("mhpy.utils.debug.torch.cuda.is_available", return_value=False)
    @patch("mhpy.utils.debug.torch.cuda.memory._record_memory_history")
    def test_enter_skips_cuda_when_unavailable(self, mock_record, mock_cuda_available):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), record_cuda=True)
            tool.__enter__()

            mock_record.assert_not_called()

    @patch("mhpy.utils.debug.torch.cuda.is_available", return_value=False)
    def test_enter_skips_cuda_when_disabled(self, mock_cuda_available):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), record_cuda=False)
            tool.__enter__()

    @patch("mhpy.utils.debug.torch.profiler.profile")
    @patch("mhpy.utils.debug.logger")
    def test_exit_exports_trace_when_profiling(self, mock_logger, mock_profile):
        mock_profiler = MagicMock()
        mock_profile.return_value = mock_profiler

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), profile=True)
            tool.__enter__()
            tool.__exit__(None, None, None)

            expected_path = str(Path(tmpdir) / "trace.json")
            mock_profiler.export_chrome_trace.assert_called_once_with(expected_path)

    @patch("mhpy.utils.debug.torch.cuda.is_available", return_value=True)
    @patch("mhpy.utils.debug.torch.cuda.memory._dump_snapshot")
    @patch("mhpy.utils.debug.torch.cuda.memory._record_memory_history")
    @patch("mhpy.utils.debug.logger")
    def test_exit_dumps_cuda_snapshot(self, mock_logger, mock_record, mock_dump, mock_cuda_available):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DebugTool(dir=Path(tmpdir), record_cuda=True)
            tool.__enter__()
            tool.__exit__(None, None, None)

            expected_path = str(Path(tmpdir) / "cuda_snapshot.pickle")
            mock_dump.assert_called_once_with(expected_path)
            # Check that recording is disabled after dump
            mock_record.assert_any_call(enabled=None)
