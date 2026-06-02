from __future__ import annotations


class TimeSeriesPreprocessor:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "Time-series feature engineering moved into the v0.4.5 Engine candidate family. "
            "Use EngineConfig(enable_time_series_functions=True)."
        )
