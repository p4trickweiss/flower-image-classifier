## Bilder einfügen

Speichere Screenshots aus den Notebooks in den `images/`-Ordner. Die benötigten Bilder:

| Dateiname | Quelle | Inhalt |
|---|---|---|
| `eda_sample_images.png` | `01_eda.ipynb` | Beispielbilder je Klasse |
| `class_distribution.png` | `01_eda.ipynb` | Klassenverteilung |
| `learning_curves_baseline.png` | `03_training.ipynb` | Lernkurven Baseline |
| `learning_curves_all.png` | `03_training.ipynb` | Lernkurven aller Modelle |
| `confusion_matrix_baseline.png` | `04_evaluation.ipynb` | Konfusionsmatrix Baseline |
| `predict_example.png` | `predict.py` / `04_evaluation.ipynb` | Beispiel-Vorhersage |
| `gradcam_correct.png` | `04_evaluation.ipynb` Abschnitt 11 | Grad-CAM korrekte Vorhersagen |
| `gradcam_errors.png` | `04_evaluation.ipynb` Abschnitt 11 | Grad-CAM Fehlklassifikationen |

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