#!/bin/bash

pandoc paper.md --from markdown --to latex --citeproc --bibliography=paper.bib --metadata link-citations=true -o paper.tex

pdflatex arXiv_paper.tex
bibtex arXiv_paper
pdflatex arXiv_paper.tex
pdflatex arXiv_paper.tex