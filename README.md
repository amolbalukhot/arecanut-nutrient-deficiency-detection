# Arecanut Leaf Verification & Nutrient Deficiency Detection

Flask web app with a **two-stage deep learning pipeline**:

1. **Stage 1 - Verification CNN** (binary, `Arecanut` / `Non_Arecanut`). Rejected images stop here; Stage 2 never runs.
2. **Stage 2 - MobileNetV2** (the only nutrient classifier): `Healthy`, `Nitrogen_Deficiency`, `Potassium_Deficiency`, `Magnesium_Deficiency` (this exact order everywhere, defined in `config.CLASS_NAMES`).

Also: Grad-CAM, fertilizer guidance (no exact doses), SQLite history, dashboard, ReportLab PDF reports.

## Model conventions
| | Verification CNN | MobileNetV2 |
|---|---|---|
| Input | 128x128x3, raw 0-255 (model rescales 1/255 internally) | 224x224x3, scaled once to [-1,1] outside the model |
| Output | sigmoid = P(Non_Arecanut); accepted if `1-p >= 0.60` (`VERIFICATION_THRESHOLD`) | softmax over `CLASS_NAMES` |
| File | `saved_models/verification_model.keras` | `saved_models/mobilenetv2_model.keras` |

Softmax scores are **not calibrated probabilities**.

## Install
Tested with Python 3.12.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set SECRET_KEY (Windows: copy .env.example .env)
```

## Datasets
```
verification_dataset/{Arecanut,Non_Arecanut}/
dataset/{Healthy,Nitrogen_Deficiency,Potassium_Deficiency,Magnesium_Deficiency}/
```
Formats: JPG, JPEG, PNG, WEBP (validated by content with Pillow). The two datasets are never mixed. Nothing is deleted or relabelled; corrupted, unsupported and duplicate files are only reported.

## Splitting (`utils/data_split.py`)
```bash
python utils/data_split.py --dataset verification
python utils/data_split.py --dataset nutrient
python utils/data_split.py --dataset all
python utils/data_split.py --dataset nutrient --groups groups.csv        # optional group metadata
python utils/data_split.py --dataset nutrient --regenerate --yes         # replace manifest explicitly
```
- **Stratified** 80/10/10, seed 42; actual counts/percentages are reported (`data_splits/reports/*.json`).
- **Duplicate-aware**: SHA-256 for exact copies (any name/extension), 64-bit dHash (Hamming <= 4) for likely near-duplicates. Duplicate groups stay in one split. Near-duplicates and label conflicts are listed for review; not all semantic duplicates can be found.
- **Group-aware**: only when you supply `groups.csv` (`path,group`, path relative to dataset root) covering every image. Otherwise the manifest's `method` states no group metadata was used - leakage between images of the same plant is then possible.
- **Manifests** (`data_splits/*_split_manifest.csv`) are reused; added/removed/modified files raise a stale-manifest error rather than silently reshuffling. Training scripts accept `--regenerate-split --yes`.
- Each class needs at least 3 valid images.

## Train
```bash
python train_verification.py   # binary verification CNN only
python train.py                # MobileNetV2 only (--finetune-epochs 5 for optional fine-tuning; --no-pretrained for offline)
python app.py                  # http://127.0.0.1:5000
```
Checkpoints are chosen by **validation loss**; the test split is used once for final evaluation. Augmentation is training-only. Results (history, plots, confusion matrix, classification report, metrics, class mapping, seed) are written to `training_results/{verification,mobilenetv2}/`. High validation/test scores do not prove generalisation to new farms, cameras or lighting. `--no-pretrained` (random init) typically needs far more data.

## App
Routes: `/` upload, `/predict`, `/dashboard`, `/history`, `/download/<id>`. The app starts without datasets or models; predictions return an error naming the missing model and the training command.
Grad-CAM (`services/gradcam_service.py`) finds the last conv layer in the loaded, nested MobileNetV2 and overlays the heatmap; it explains model focus and is not proof of deficiency. Recommendations are general guidance; confirm with soil/leaf tests and an agronomist.

## Tests
```bash
python -m pytest -q
```
Uses mocked models and untrained MobileNetV2 (no downloads).

## Security notes
Content-validated uploads, 16 MB limit, random server-side filenames, CSRF (Flask-WTF), parameterised SQL, no stack traces shown. Set a strong `SECRET_KEY`; serve with a production WSGI server (e.g. gunicorn/waitress), not the dev server.

## Troubleshooting
- *Model missing*: run the command shown in the error.
- *Stale manifest*: dataset changed; regenerate deliberately with `--regenerate-split --yes`.
- *ImageNet weights download fails*: use `--no-pretrained` (expect lower accuracy).
- *Charts/icons not loading*: pages load Bootstrap, icons, fonts and Chart.js from CDNs and need internet access.
