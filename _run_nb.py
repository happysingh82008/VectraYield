import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

nb_path = Path(r"c:\Users\happy\VectraYield\VectraYield.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
ns = {"np": np, "pd": pd}

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    buf = io.StringIO()
    outputs = []
    try:
        with redirect_stdout(buf):
            exec(compile(src, f"cell_{i}", "exec"), ns)
        text = buf.getvalue()
        if text:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text.splitlines(keepends=True) or [text],
                }
            )
        cell["execution_count"] = i + 1
        cell["outputs"] = outputs
        print(f"OK cell {i}")
        if text.strip():
            print(text[:500])
    except Exception:
        print(f"FAIL cell {i}")
        traceback.print_exc()
        raise

# Verify
print("FINAL missing", int(ns["vectrayield_capped"].isnull().sum().sum()))
print("FINAL shape", ns["vectrayield_capped"].shape)
print("FINAL outlier cols after", ns["outlier_cols_after"])

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# Mirror executed notebook to the other copies
for dest in [
    Path(r"c:\Users\happy\VectraYield\VectraYield\VectraYield.ipynb"),
    Path(r"c:\Users\happy\OneDrive\dataset\VectraYield.ipynb"),
]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("copied executed notebook to", dest)
