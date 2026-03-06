"""Central configuration for kidney donation eligibility analysis."""

from pathlib import Path

# Paths -- resolved relative to the source tree.
# _PACKAGE_DIR = src/kidney/  ->  .parent = src/  ->  .parent = project root
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "NHIS_CSV"
NHANES_DIR = PROJECT_ROOT / "data" / "nhanes_data"

# Thresholds
OBESITY_THRESHOLD = 30.0
ELIGIBILITY_BMI_THRESHOLD = 35.0
US_ADULT_POPULATION_2025 = 260_000_000

# NHIS column mappings -- maps raw variable names to standardised names per year.
_COLS_2015_2018_BASE = {
    "AWEIGHTP": "Weight",
    "AHEIGHT": "Height",
    "AGE_P": "age",
    "REGION": "region",
    "WTFA_SA": "survey_weight",
    "HYPYR1": "hypertension",
    "HYPEV": "historic_hypertension",
}

_COLS_2019_PLUS_BASE = {
    "WEIGHTLBTC_A": "Weight",
    "HEIGHTTC_A": "Height",
    "AGEP_A": "age",
    "REGION": "region",
    "DIBPILL_A": "diabetes",
    "PREDIB_A": "prediabetes",
    "HYP12M_A": "hypertension",
    "HYPEV_A": "historic_hypertension",
    "NOTCOV_A": "insurance",
    "POVRATTC_A": "poverty_ratio",
    "HISPALLP_A": "race_ethnicity",
    "WTFA_A": "survey_weight",
}

COLUMN_MAPPINGS: dict[str, dict[str, str]] = {
    "2015": {**_COLS_2015_2018_BASE, "DIBPILL": "diabetes", "DIBPRE1": "prediabetes"},
    "2016": {**_COLS_2015_2018_BASE, "DIBPILL1": "diabetes", "DIBPRE2": "prediabetes"},
    "2017": {**_COLS_2015_2018_BASE, "DIBPILL1": "diabetes", "DIBPRE2": "prediabetes"},
    "2018": {**_COLS_2015_2018_BASE, "DIBPILL1": "diabetes", "DIBPRE2": "prediabetes"},
    "2019": {**_COLS_2019_PLUS_BASE},
    "2020": {**_COLS_2019_PLUS_BASE},
    "2021": {**_COLS_2019_PLUS_BASE, "EDUCP_A": "education"},
    "2022": {**_COLS_2019_PLUS_BASE, "EDUCP_A": "education"},
    "2023": {**_COLS_2019_PLUS_BASE, "EDUCP_A": "education"},
    "2024": {**_COLS_2019_PLUS_BASE, "EDUCP_A": "education"},
}

FILE_PATHS: dict[str, Path] = {
    "2015": DATA_DIR / "adult15csv" / "samadult.csv",
    "2016": DATA_DIR / "adult16csv" / "samadult.csv",
    "2017": DATA_DIR / "adult17csv" / "samadult.csv",
    "2018": DATA_DIR / "adult18csv" / "samadult.csv",
    "2019": DATA_DIR / "adult19csv" / "adult19.csv",
    "2020": DATA_DIR / "adult20csv" / "adult20.csv",
    "2021": DATA_DIR / "adult21csv" / "adult21.csv",
    "2022": DATA_DIR / "adult22csv" / "adult22.csv",
    "2023": DATA_DIR / "adult23csv" / "adult23.csv",
    "2024": DATA_DIR / "adult24csv" / "adult24.csv",
}

# Region mappings
REGIONS: dict[int, str] = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

REGION_COLORS: dict[str, str] = {
    "Northeast": "#E63946",
    "Midwest": "#1D3557",
    "South": "#2A9D8F",
    "West": "#F4A261",
}

# Age groups
AGE_GROUPS: list[tuple[int, int]] = [(18, 30), (31, 45), (46, 65), (66, 84)]
AGE_GROUP_LABELS: list[str] = ["18-30", "31-45", "46-65", "66-84"]

# Insurance status (NOTCOV_A: 1=uninsured, 2=covered)
INSURANCE_STATUS: dict[int, str] = {1: "Not Covered", 2: "Covered"}

# Poverty ratio categories
POVERTY_CATEGORIES: dict[str, tuple[float, float]] = {
    "extreme_poverty": (0.0, 1.0),
    "low_income": (1.0, 2.0),
    "moderate_income": (2.0, 4.0),
    "higher_income": (4.0, 11.01),
}
POVERTY_CATEGORY_LABELS: list[str] = ["<1x FPL", "1-2x FPL", "2-4x FPL", "4x+ FPL"]

# Race/ethnicity
RACE_ETHNICITY_CODES: dict[int, str] = {
    1: "Hispanic",
    2: "Non-Hispanic White",
    3: "Non-Hispanic Black",
    4: "Non-Hispanic Asian",
}
RACE_ETHNICITY_COLORS: dict[str, str] = {
    "Hispanic": "#e41a1c",
    "Non-Hispanic White": "#377eb8",
    "Non-Hispanic Black": "#4daf4a",
    "Non-Hispanic Asian": "#984ea3",
}

# Education categories (EDUCP_A code ranges, half-open)
EDUCATION_CATEGORIES: dict[str, tuple[int, int]] = {
    "less_than_hs": (0, 3),
    "hs_graduate": (3, 5),
    "some_college": (5, 8),
    "bachelors_plus": (8, 11),
}
EDUCATION_LABELS: list[str] = [
    "Less than HS",
    "HS Graduate/GED",
    "Some College/Associate",
    "Bachelor's or Higher",
]
EDUCATION_COLORS: dict[str, str] = {
    "Less than HS": "#d73027",
    "HS Graduate/GED": "#fc8d59",
    "Some College/Associate": "#fee08b",
    "Bachelor's or Higher": "#1a9850",
}

# Condition and plot colors
CONDITION_COLORS: dict[str, str] = {
    "Diabetes": "#FF9999",
    "Hypertension": "#66B2FF",
    "Obesity": "#99FF99",
    "Prediabetes": "#FFCC99",
}
PLOT_COLORS: list[str] = [
    "#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FF99CC", "#99CCFF",
]

# NHANES file paths
DEFAULT_NHANES_FILES: dict[str, Path] = {
    "demo": NHANES_DIR / "DEMO_L.xpt",
    "diq": NHANES_DIR / "DIQ_L.xpt",
    "bpq": NHANES_DIR / "BPQ_L.xpt",
    "bpx": NHANES_DIR / "BPXO_L.xpt",
    "bmx": NHANES_DIR / "BMX_L.xpt",
    "ghb": NHANES_DIR / "GHB_L.xpt",
    "glu": NHANES_DIR / "GLU_L.xpt",
}
NHANES_FILE_ORDER: tuple[str, ...] = ("demo", "diq", "bpq", "bpx", "bmx", "ghb", "glu")
NHANES_WEIGHT_COLUMNS: tuple[str, ...] = ("WTSAF2YR", "WTSAF2Y_L", "WTSAF2Y", "WTMEC2YR")
NHANES_CONDITION_ORDER: list[str] = [
    "Diabetes", "Prediabetes", "Hypertension",
    "Historic Hypertension", "Obesity", "Any Condition",
]
