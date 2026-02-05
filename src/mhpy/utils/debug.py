from contextlib import nullcontext
from pathlib import Path

from loguru import logger
import torch

from mhpy.utils.common import get_run_dir


class DebugTool:
    def __init__(self, profile=False, record_cuda=False, profile_activities=None, cuda_max_entries=100000, log_dir: Path | None = None):
        self.log_dir = log_dir or get_run_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.record_cuda = record_cuda
        self.cuda_max_entries = cuda_max_entries

        self.profile = profile
        self.prof_context = (
            torch.profiler.profile(
                activities=profile_activities or [torch.profiler.ProfilerActivity.CPU],
                with_stack=True,
                profile_memory=True,
            )
            if self.profile
            else nullcontext()
        )

    def __enter__(self):
        self.prof_context.__enter__()
        if self.record_cuda and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=self.cuda_max_entries)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prof_context.__exit__(exc_type, exc_val, exc_tb)

        if self.profile:
            trace_path = self.log_dir / "trace.json"
            self.prof_context.export_chrome_trace(str(trace_path))
            logger.info(f"Trace exported to {trace_path}")

        if self.record_cuda and torch.cuda.is_available():
            mem_path = self.log_dir / "cuda_snapshot.pickle"
            try:
                torch.cuda.memory._dump_snapshot(str(mem_path))
                torch.cuda.memory._record_memory_history(enabled=None)
                logger.info(f"CUDA Snapshot exported to {mem_path}")
            except Exception as e:
                logger.error(f"Failed to dump CUDA snapshot: {e}")
