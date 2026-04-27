# 🌸 flower-image-classifier

End-to-end ML pipeline for classifying flower species from images — covering data preprocessing, model training, evaluation, and inference. Built for an AI Engineering university course.

## Project Structure

```
flower-image-classifier/
├── configs/          # experiment configuration files
├── data/             # raw and processed images (not tracked in git)
├── models/           # saved model checkpoints (not tracked in git)
├── notebooks/        # EDA, preprocessing, training, and evaluation notebooks
├── src/              # source code
```

## Setup

```bash
git clone https://github.com/p4trickweiss/flower-image-classifier.git
cd flower-image-classifier
pip install -r requirements.txt
```

## Training on the AI Server

Clone the repo and set up the GPU environment:

```bash
git clone https://github.com/p4trickweiss/flower-image-classifier.git
cd flower-image-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
```

Check which GPUs are available with `nvidia-smi`, then run training targeting a free GPU (e.g. GPU 1):

```bash
CUDA_VISIBLE_DEVICES=1 python src/train.py
```

Training saves two files to `models/`:
- `best_model.keras` — best checkpoint by val accuracy
- `history.json` — loss and accuracy per epoch for plotting

## Downloading the Model Locally

After training, copy the outputs to your local machine from the project root:

```bash
scp <user>@aiserver:/home/<user>/flower-image-classifier/models/best_model.keras ./models/
scp <user>@aiserver:/home/<user>/flower-image-classifier/models/history.json ./models/
```

Then open `notebooks/03_training.ipynb` to plot the learning curves.

## Inference

Run prediction on a single image:

```bash
python src/predict.py /path/to/image.jpg
```

## Data


The [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers]tf_flowers) dataset was chosen for its practical suitability for training a CNN from scratch. With 5 well-defined classes (daisy, dandelion, rose, sunflower, tulip) and approximately 3,600 images, the dataset is large enough to train a meaningful model while remaining computationally manageable. Since pretrained models are not permitted in this project, a dataset with a tractable number of classes improves the likelihood of achieving competitive accuracy without transfer learning.

