"""Spring–mass coarsening (downsampling) utilities and CLI."""

from __future__ import annotations

from downsampling.validate import (
    DownsampledBundleError,
    assert_coarse_final_data_loads_in_realdata,
    assert_realdata_layout,
    validate_downsampled_bundle,
)

__all__ = [
    "DownsampledBundleError",
    "assert_coarse_final_data_loads_in_realdata",
    "assert_realdata_layout",
    "validate_downsampled_bundle",
]
