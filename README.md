# 🌸 flower-image-classifier

End-to-end ML pipeline for classifying flower species from images — covering data preprocessing, model training, evaluation, and inference. Built for the AI Engineering university course.

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
git clone https://github.com/<your-username>/flower-image-classifier.git
cd flower-image-classifier
pip install -r requirements.txt
```

## Data


The [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers]tf_flowers) dataset was chosen for its practical suitability for training a CNN from scratch. With 5 well-defined classes (daisy, dandelion, rose, sunflower, tulip) and approximately 3,600 images, the dataset is large enough to train a meaningful model while remaining computationally manageable. Since pretrained models are not permitted in this project, a dataset with a tractable number of classes improves the likelihood of achieving competitive accuracy without transfer learning.

## Course

AI Engineering — Hochschule Campus Wien