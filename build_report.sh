#!/bin/bash
# Build LaTeX report (pdflatex + bibtex, 3 passes)
# Usage: ./build_report.sh [midterm|final]
#   default: midterm

TYPE="${1:-midterm}"

if [ "$TYPE" = "midterm" ]; then
    TEX="midterm_report430"
elif [ "$TYPE" = "final" ]; then
    TEX="report430"
else
    echo "Usage: ./build_report.sh [midterm|final]"
    exit 1
fi

cd "$(dirname "$0")/report" || exit 1

echo "Building $TEX.tex ..."
pdflatex -interaction=nonstopmode "$TEX.tex" > /dev/null 2>&1
bibtex "$TEX" > /dev/null 2>&1
pdflatex -interaction=nonstopmode "$TEX.tex" > /dev/null 2>&1
pdflatex -interaction=nonstopmode "$TEX.tex" > /dev/null 2>&1

if [ -f "$TEX.pdf" ]; then
    echo "Done: report/$TEX.pdf"
else
    echo "Build failed. Check report/$TEX.log"
    exit 1
fi
