import json
from copy import deepcopy
from pathlib import Path


def src_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    # Jupyter often stores the last line without a trailing newline
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def make_code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src_lines(text),
    }


def make_md(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src_lines(text),
    }


LOAD_CELL = '''import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

dataset_candidates = [
    Path(r"C:\\Users\\happy\\OneDrive\\dataset\\uci-secom.csv"),
    Path("uci-secom.csv"),
]
csv_path = next((path for path in dataset_candidates if path.exists()), None)
if csv_path is None:
    raise FileNotFoundError("uci-secom.csv not found in dataset folder or project root")

print("Loading VectraYield from:", csv_path)
vectrayield = pd.read_csv(csv_path)
vectrayield.sample(10)'''

HIGH_NULL_CELL = '''vectrayield_HighNulls = vectrayield_missing[vectrayield_missing >= 0.50]
print("Columns with missing values >= 50%")
print(vectrayield_HighNulls)
print("Count:", len(vectrayield_HighNulls))'''

MD_CELL = '''## Missing values and outliers

1. Drop sensors that are at least 50% empty, then fill remaining numeric nulls with `SimpleImputer(strategy="median")`.
2. Handle outliers with **IQR capping** (winsorize at 1.5 IQR). Row trimming is not used: an outlier in any column would drop every wafer.'''

DROP_CELL = '''vectrayield_clean = vectrayield.drop(columns=vectrayield_HighNulls.index, errors="ignore")
print("Shape after dropping high-null columns:", vectrayield_clean.shape)
print("Missing values before SimpleImputer:", int(vectrayield_clean.isnull().sum().sum()))'''

IMPUTE_CELL = '''target_col = "Pass/Fail"
exclude = {target_col}
if "Time" in vectrayield_clean.columns:
    exclude.add("Time")

numeric_cols = [
    col for col in vectrayield_clean.columns
    if col not in exclude and pd.api.types.is_numeric_dtype(vectrayield_clean[col])
]

imputer = SimpleImputer(strategy="median")
vectrayield_imputed = vectrayield_clean.copy()
vectrayield_imputed[numeric_cols] = imputer.fit_transform(vectrayield_clean[numeric_cols])

print("Numeric feature columns:", len(numeric_cols))
print("Missing values after SimpleImputer:", int(vectrayield_imputed.isnull().sum().sum()))'''

CAP_CELL = '''def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_column_count(df, cols):
    count = 0
    for col in cols:
        lower, upper = iqr_bounds(df[col])
        if ((df[col] < lower) | (df[col] > upper)).any():
            count += 1
    return count


outlier_cols_before = outlier_column_count(vectrayield_imputed, numeric_cols)

vectrayield_capped = vectrayield_imputed.copy()
for col in numeric_cols:
    lower, upper = iqr_bounds(vectrayield_imputed[col])
    vectrayield_capped[col] = vectrayield_imputed[col].clip(lower=lower, upper=upper)

outlier_cols_after = outlier_column_count(vectrayield_capped, numeric_cols)

print("Shape after IQR capping:", vectrayield_capped.shape)
print("Columns with IQR outliers before capping:", outlier_cols_before)
print("Columns with IQR outliers after capping:", outlier_cols_after)
print("Remaining missing values:", int(vectrayield_capped.isnull().sum().sum()))
vectrayield_capped.head()'''


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_or_set(cells: list, pred, new_cell: dict) -> bool:
    for i, cell in enumerate(cells):
        if pred(cell):
            cells[i] = new_cell
            return True
    return False


def strip_trailing_empty(cells: list) -> list:
    while cells and cell_text(cells[-1]).strip() == "":
        cells.pop()
    return cells


def already_has_imputer(cells: list) -> bool:
    return any("SimpleImputer" in cell_text(c) and "fit_transform" in cell_text(c) for c in cells)


def update_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Keep original load cell outputs if present; still replace source so sklearn is imported.
    load_src = cell_text(cells[0]) if cells else ""
    cells[0]["source"] = src_lines(LOAD_CELL)
    if "SimpleImputer" not in load_src:
        cells[0]["outputs"] = []
        cells[0]["execution_count"] = None

    for cell in cells:
        if "vectrayield_HighNulls" in cell_text(cell) and "vectrayield_missing[" in cell_text(cell):
            cell["source"] = src_lines(HIGH_NULL_CELL)
            if ">=50" in cell_text(cell) or ">= 50" in "".join(cell.get("source", [])):
                pass
            if ">= 0.50" not in "".join(cell.get("source", [])) and ">=0.50" not in cell_text(cell):
                cell["outputs"] = []
                cell["execution_count"] = None

    # Force high-null cell rewrite by matching more loosely
    for cell in cells:
        txt = cell_text(cell)
        if "vectrayield_HighNulls" in txt and "vectrayield_missing" in txt and "drop" not in txt:
            cell["source"] = src_lines(HIGH_NULL_CELL)

    cells = strip_trailing_empty(cells)

    # Remove previous contribution cells so we can insert a clean set
    cleaned = []
    skip_ids = {
        "contrib-impute-md",
        "contrib-drop-high-nulls",
        "contrib-simple-imputer",
        "contrib-iqr-cap",
        "missing-outlier-md",
        "drop-high-nulls",
        "simple-imputer",
        "iqr-cap",
    }
    for cell in cells:
        txt = cell_text(cell)
        if cell.get("id") in skip_ids:
            continue
        if txt.startswith("vectrayield_clean = vectrayield.drop"):
            continue
        if "SimpleImputer(strategy=" in txt and "fit_transform" in txt:
            continue
        if txt.startswith("def iqr_cap(") or txt.startswith("def iqr_bounds("):
            continue
        if txt.startswith("## Missing values") or txt.startswith("## Contribution"):
            continue
        cleaned.append(cell)

    cleaned.extend(
        [
            make_md("missing-outlier-md", MD_CELL),
            make_code("drop-high-nulls", DROP_CELL),
            make_code("simple-imputer", IMPUTE_CELL),
            make_code("iqr-cap", CAP_CELL),
        ]
    )

    nb["cells"] = cleaned
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", path, "cells", len(cleaned))


paths = [
    Path(r"c:\Users\happy\VectraYield\VectraYield.ipynb"),
    Path(r"c:\Users\happy\VectraYield\VectraYield\VectraYield.ipynb"),
    Path(r"c:\Users\happy\OneDrive\dataset\VectraYield.ipynb"),
]
for p in paths:
    if p.exists():
        update_notebook(p)
    else:
        print("missing", p)
