"""Original project manuscript, page previews and explicitly attributed highlights."""
import json

from pulmotrack.archive import ROOT

PAPER_ROOT = ROOT / "assets" / "paper"


def load_paper():
    return json.loads((PAPER_ROOT / "manifest.json").read_text(encoding="utf-8"))


def paper_path(filename):
    path = (PAPER_ROOT / filename).resolve()
    if path.parent != PAPER_ROOT.resolve() or not path.is_file():
        raise ValueError("The document asset is outside the research paper collection.")
    return path
