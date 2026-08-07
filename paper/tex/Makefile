# Build all paper drafts. Requires pdflatex and bibtex.
#   make          build every PDF
#   make clean    remove build artifacts, keep the PDFs
#   make purge    remove PDFs too
PAPERS := benchmark_paper segmentation detection
PDFS   := $(addsuffix .pdf,$(PAPERS))

all: $(PDFS)

%.pdf: %.tex references.bib
	pdflatex -interaction=nonstopmode $*.tex >/dev/null
	bibtex $* >/dev/null 2>&1 || true
	pdflatex -interaction=nonstopmode $*.tex >/dev/null
	pdflatex -interaction=nonstopmode $*.tex >/dev/null
	@echo "built $@"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc

purge: clean
	rm -f $(PDFS)

.PHONY: all clean purge
