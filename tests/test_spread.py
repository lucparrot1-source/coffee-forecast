import numpy as np
import pandas as pd
import pytest


def _wide(kc: list[float], rm: list[float]) -> pd.DataFrame:
    """Helper: build a wide monthly price DataFrame from two lists."""
    dates = pd.date_range("2020-01-01", periods=len(kc), freq="MS")
    return pd.DataFrame({"KC=F": kc, "RM=F": rm}, index=dates)
