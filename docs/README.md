# Dokumentation – Blumenerkennung mit CNNs

## Inhalt

- `report.md` — Hauptbericht (Markdown)
- `images/` — Bilder für den Bericht (Screenshots aus Notebooks einfügen)

## Bilder einfügen

Speichere Screenshots aus den Notebooks in den `images/`-Ordner. Die benötigten Bilder sind im Report bereits referenziert:

| Dateiname | Quelle | Inhalt |
|---|---|---|
| `eda_sample_images.png` | `01_eda.ipynb` | Beispielbilder je Klasse |
| `class_distribution.png` | `01_eda.ipynb` | Klassenverteilung |
| `learning_curves_baseline.png` | `03_training.ipynb` | Lernkurven Baseline |
| `learning_curves_all.png` | `03_training.ipynb` | Lernkurven aller Modelle |
| `confusion_matrix_baseline.png` | `04_evaluation.ipynb` | Konfusionsmatrix Baseline |
| `predict_example.png` | `predict.py` / `04_evaluation.ipynb` | Beispiel-Vorhersage |

## Markdown → PDF konvertieren

### Option 1: Pandoc (empfohlen)

```bash
# Installation (macOS)
brew install pandoc
brew install --cask basictex  # oder: brew install mactex

# PDF erzeugen
cd docs
pandoc report.md -o report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  -V fontsize=11pt \
  -V lang=de \
  --highlight-style=tango
```

### Option 2: VS Code Extension

1. Extension installieren: **Markdown PDF** (yzane.markdown-pdf)
2. `report.md` öffnen
3. Rechtsklick → "Markdown PDF: Export (pdf)"

### Option 3: Typora

Typora (kostenpflichtig) rendert Markdown direkt und kann als PDF exportieren.
