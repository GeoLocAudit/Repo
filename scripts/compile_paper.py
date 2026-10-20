"""Compile paper/main.tex into paper/build/main.pdf (pdflatex, bibtex, pdflatex x2).

  python scripts/compile_paper.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.common import ROOT  # noqa: E402


def main():
    paper, build = ROOT / 'paper', ROOT / 'paper' / 'build'
    build.mkdir(exist_ok=True)
    (build / 'references.bib').write_bytes((paper / 'references.bib').read_bytes())
    run = lambda *c: subprocess.run(list(c), cwd=paper, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run('pdflatex', '-interaction=nonstopmode', '-output-directory=build', 'main.tex')
    subprocess.run(['bibtex', 'main'], cwd=build, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(2):
        run('pdflatex', '-interaction=nonstopmode', '-output-directory=build', 'main.tex')
    print('paper compiled -> paper/build/main.pdf')


if __name__ == '__main__':
    main()
