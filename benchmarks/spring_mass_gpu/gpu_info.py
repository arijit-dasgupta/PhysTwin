"""GPU identity + sanity checks for the benchmark report."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import torch


@dataclass
class GpuInfoReport:
    device_index: int
    device_name: str
    properties_total_memory_bytes: int
    mem_get_info_total_bytes: int
    mem_get_info_free_bytes: int
    sanity_total_mem_match: bool
    nvidia_smi_line: str | None
    cuda_version: str | None


def collect_gpu_report(device_index: int = 0) -> GpuInfoReport:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; benchmarks require a GPU.")

    name = torch.cuda.get_device_name(device_index)
    props = torch.cuda.get_device_properties(device_index)
    total_props = int(props.total_memory)
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    # Allow small driver accounting differences
    match = abs(total_b - total_props) / max(total_props, 1) < 0.02

    smi: str | None = None
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        smi = None

    cv = getattr(torch.version, "cuda", None)

    return GpuInfoReport(
        device_index=device_index,
        device_name=name,
        properties_total_memory_bytes=total_props,
        mem_get_info_total_bytes=int(total_b),
        mem_get_info_free_bytes=int(free_b),
        sanity_total_mem_match=match,
        nvidia_smi_line=smi,
        cuda_version=str(cv) if cv is not None else None,
    )
