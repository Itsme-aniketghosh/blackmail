"""Pull the embedded PNGs out of the executed notebook into figures/.

The run wrote its figures to Drive, which is not in the repo. But Jupyter also embeds every
displayed image in the .ipynb as base64, so the executed notebook is a complete record of
what was actually produced. Extracting from there rather than asking for the Drive files
means the figure in the report is provably the one the run made.
"""
import base64
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
NB = REPO / "notebooks" / "runs" / "05_function_vector.ipynb"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

nb = json.loads(NB.read_text(encoding="utf-8"))

# Cell title -> the name the report will use. Anything not listed still gets written, under a
# fallback name, so a new figure never silently goes missing.
NAMES = {
    "CELL 12 ": "01_stack",
    "CELL 12b": "02_coherence_walkback",
    "CELL 15 ": "03_delivery",
    "CELL 17L": "04_transfer_STALE",
}

found = []
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    title = next((l for l in src.splitlines() if l.startswith("# CELL")), "")
    key = next((k for k in NAMES if title.startswith("# " + k.strip())), None)
    # match on the exact cell tag, e.g. "# CELL 12b —" vs "# CELL 12 —"
    tag = re.match(r"# (CELL [\w]+)", title)
    key = None
    if tag:
        for k in NAMES:
            if tag.group(1).strip() == "CELL " + k.replace("CELL ", "").strip():
                key = k
                break
    for j, o in enumerate(c.get("outputs", [])):
        png = o.get("data", {}).get("image/png")
        if not png:
            continue
        stem = NAMES.get(key, f"cell{i}_out{j}")
        if len([f for f in found if f[0] == stem]):
            stem = f"{stem}_{j}"
        raw = base64.b64decode(png if isinstance(png, str) else "".join(png))
        p = OUT / f"{stem}.png"
        p.write_bytes(raw)
        found.append((stem, len(raw), i, title[:58]))

print(f"{len(found)} figures -> {OUT}")
for stem, n, i, t in found:
    print(f"  {stem+'.png':32s} {n/1024:7.1f} KB   from cell {i}: {t}")

if not found:
    print("no embedded images. The notebook was saved without outputs, or the figures were "
          "written to Drive and never displayed inline.")
