"""Linear trend fitting and projection utilities."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from kidney.analysis.prevalence import weighted_prevalence
from kidney.config import ELIGIBILITY_BMI_THRESHOLD, OBESITY_THRESHOLD


def fit_linear_trend(
    years: list[str],
    values: list[float],
    fit_start_year: int = 2019,
) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    """Fit OLS to data from *fit_start_year* onward.

    Returns (model, fit_years_int_array, fit_values_array).
    """
    idx = [i for i, y in enumerate(years) if int(y) >= fit_start_year]
    if not idx:
        raise ValueError(f"No data from year {fit_start_year}")
    yrs = np.array([int(years[i]) for i in idx])
    vals = np.array([values[i] for i in idx])
    base = yrs[0]
    model = LinearRegression().fit((yrs - base).reshape(-1, 1), vals)
    return model, yrs, vals


def project_trend(
    model: LinearRegression, base_year: int, projection_years: list[int]
) -> np.ndarray:
    """Project fitted model to future years."""
    X = (np.array(projection_years) - base_year).reshape(-1, 1)
    return model.predict(X)


def trend_p_values(dataframes: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute p-values for linear trends across all years."""
    years = sorted(dataframes)
    year_nums = np.array([int(y) for y in years])
    results: dict[str, float] = {}

    for condition in ("diabetes", "prediabetes", "hypertension", "obesity", "eligible"):
        prevs = []
        for year in years:
            df = dataframes[year]
            if condition == "obesity":
                prevs.append(weighted_prevalence(df, df["BMI"] > OBESITY_THRESHOLD))
            elif condition == "eligible":
                mask = (
                    (df["diabetes"] != 1)
                    & (df["hypertension"] != 1)
                    & (df["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
                )
                prevs.append(weighted_prevalence(df, mask))
            else:
                prevs.append(weighted_prevalence(df, df[condition] == 1))
        _, _, _, p, _ = stats.linregress(year_nums, prevs)
        results[condition] = float(p)
    return results
