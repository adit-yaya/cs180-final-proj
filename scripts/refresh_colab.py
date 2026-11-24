"""Refresh the standalone notebook's bundled source after changing the demo."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Generative_Zoom.ipynb"
FILES = (
    "app.py", "director.py", "generator.py", "imaging.py",
    "static/index.html", "static/app.js", "requirements.txt",
    "requirements-model.txt", "requirements-colab.txt",
)


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    bundle = next(cell for cell in notebook["cells"] if cell["id"] == "project-files")
    lines = ["#@title Unpack the bundled demo files\n", "from pathlib import Path\n",
             "from tempfile import mkdtemp\n", "project = Path(mkdtemp(prefix='generative-zoom-'))\n",
             "files = {\n"]
    for name in FILES:
        lines.append(f"    {name!r}: {(ROOT / name).read_text()!r},\n")
    lines.extend(["}\n", "for name, source in files.items():\n",
                  "    path = project / name\n", "    path.parent.mkdir(exist_ok=True)\n",
                  "    path.write_text(source, encoding='utf-8')\n"])
    bundle["source"] = lines
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
