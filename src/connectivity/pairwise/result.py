"""Common result containers for connectivity estimators."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import xarray as xr


@dataclass
class MetricResult:
    metric: str
    pair_table: pd.DataFrame
    detail: xr.Dataset
    auxiliary_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    runtime_metadata: dict[str, object] = field(default_factory=dict)
