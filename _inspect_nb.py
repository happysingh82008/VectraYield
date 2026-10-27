import json
from pathlib import Path

for p in [
    Path(r"c:\Users\happy\VectraYield\VectraYield.ipynb"),
    Path(r"c:\Users\happy\VectraYield\VectraYield\VectraYield.ipynb"),
    Path(r"c:\Users\happy\OneDrive\dataset\VectraYield.ipynb"),
]:
    print("=" * 80)
    print(p, "exists" if p.exists() else "MISSING")
    if not p.exists():
        continue
    nb = json.loads(p.read_text(encoding="utf-8"))
    print("n_cells", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        print(f"\n--- {i} {c['cell_type']} ---")
        print(src[:1500] if src else "<empty>")
