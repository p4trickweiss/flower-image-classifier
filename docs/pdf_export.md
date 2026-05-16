## Markdown → PDF konvertieren

```bash
# Installation (macOS)
brew install pandoc
brew install --cask basictex

# PDF erzeugen
cd docs
pandoc report.md -o report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  -V fontsize=11pt \
  -V lang=de \
  --syntax-highlighting=tango
```