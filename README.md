# 🌸 flower-image-classifier

End-to-end ML pipeline for classifying flower species from images — covering data preprocessing, model training, evaluation, and inference.

## Project Documentation Report

The detailed project documentation report can be found [here](./docs/report.md).

## Data


The [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers]tf_flowers) dataset was chosen for its practical suitability for training a CNN from scratch. With 5 well-defined classes (daisy, dandelion, rose, sunflower, tulip) and approximately 3,600 images, the dataset is large enough to train a meaningful model while remaining computationally manageable. Since pretrained models are not permitted in this project, a dataset with a tractable number of classes improves the likelihood of achieving competitive accuracy without transfer learning.

## Project Structure

```
flower-image-classifier/
├── configs/          # experiment configuration files
├── data/             # raw and processed images (not tracked in git)
├── models/           # saved model checkpoints (not tracked in git)
├── notebooks/        # EDA, preprocessing, training, and evaluation notebooks
├── src/              # source code
```
The contents of the model directory can be downloaded [here](https://drive.google.com/file/d/1xapZKP61CehDycHj_J41R1B0T-c1JHUQ/view?usp=sharing)  
The contents of the data directory can be downloaded [here](https://drive.google.com/file/d/1y4V5lbc11C4jkz8SbFk3PuVGp78Nw69e/view?usp=sharing)

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
CUDA_VISIBLE_DEVICES=1 python src/train.py configs/baseline.yaml
```

A config file can be passed as an optional argument (defaults to `configs/baseline.yaml`). The available configs are:

| Config | Optimizer | Notes |
|---|---|---|
| `configs/baseline.yaml` | Adam | default |
| `configs/shallow_sgd.yaml` | SGD | fewer layers |
| `configs/wide_rmsprop.yaml` | RMSprop | wider filters |
| `configs/alexnet.yaml` | Adam | AlexNet architecture, 224×224 input |
| `configs/cnn_svm.yaml` | — | baseline CNN feature extractor + SVM head |

The hybrid CNN+SVM model is trained separately using `src/train_hybrid.py`:

```bash
python src/train_hybrid.py configs/cnn_svm.yaml
```

This extracts 512-dim embeddings from the baseline CNN backbone and trains an SVM classifier on top. It saves `models/cnn_svm_classifier.pkl` (the SVM) and reuses the existing `models/baseline_best_model.keras` as the feature extractor.

Training saves two files to `models/`, named after the config:
- `{config}_best_model.keras` — best checkpoint by val accuracy
- `{config}_history.json` — loss and accuracy per epoch for plotting

## Downloading the Model Locally

After training, copy the outputs to your local machine from the project root:

```bash
scp <user>@aiserver:/home/<user>/flower-image-classifier/models/baseline_best_model.keras ./models/
scp <user>@aiserver:/home/<user>/flower-image-classifier/models/baseline_history.json ./models/
```

Then open `notebooks/03_training.ipynb` to plot the learning curves.

## Inference

Run prediction on a single image:

```bash
# with default config (baseline)
python src/predict.py /path/to/image.jpg

# with a specific config/model
python src/predict.py /path/to/image.jpg configs/shallow_sgd.yaml
```

## Evaluation on Custom Test Images

To evaluate the model on your own labeled images, place them in class subfolders:

```
data/custom_test/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

Then run:

```bash
python src/evaluate.py configs/baseline.yaml data/custom_test
```

This prints per-class precision/recall/F1 and a confusion matrix.

## Results

### Model Comparison
| Model           | Params | Test Acc | Test Loss | Own Images Acc | Macro F1 |
|-----------------|--------|----------|-----------|----------------|----------|
| baseline.yaml   | 8.78M  | 0.8420   | 0.4577    | 0.79           | 0.79     |
| shallow_sgd.yaml| 16.80M | 0.7575   | 0.7070    | 0.61           | 0.60     |
| wide_rmsprop.yaml| 18.34M | 0.7575   | 0.6159    | 0.69           | 0.67     |
| alexnet.yaml    | 58.30M | 0.8283   | 0.4361    | 0.69           | 0.69     |
| cnn_svm.yaml     | 8.78M  | 0.8147   | —         | 0.77           | 0.77     |

### Detailed Test Results
#### Baseline.yaml
epochs: 50  
batch_size: 32  
optimizer: adam  
learning_rate: 0.001  
image_size: 128  
num_blocks: 4  
filters_start: 32  
dropout_rate: 0.5  
dense_units: 512

Total params: 8,782,021 (33.50 MB)  
Trainable params: 8,781,061 (33.50 MB)  
Non-trainable params: 960 (3.75 KB)  

Test accuracy : 0.8420  
Test loss     : 0.4577

#### Test with custom test-images (data/test-images)
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Dandelion  | 0.58      | 0.87   | 0.69     | 30      |
| Daisy      | 1.00      | 0.83   | 0.91     | 30      |
| Tulips     | 0.83      | 0.50   | 0.62     | 30      |
| Sunflowers | 0.83      | 0.80   | 0.81     | 30      |
| Roses      | 0.85      | 0.93   | 0.89     | 30      |
| **Accuracy**   |           |        | **0.79** | **150** |
| Macro Avg  | 0.82      | 0.79   | 0.79     | 150     |
| Weighted Avg | 0.82    | 0.79   | 0.79     | 150     |

| True \ Predicted | Dandelion | Daisy | Tulips | Sunflowers | Roses |
|------------------|-----------|-------|--------|------------|-------|
| **Dandelion**    | 26        | 0     | 1      | 3          | 0     |
| **Daisy**        | 5         | 25    | 0      | 0          | 0     |
| **Tulips**       | 8         | 0     | 15     | 2          | 5     |
| **Sunflowers**   | 6         | 0     | 0      | 24         | 0     |
| **Roses**        | 0         | 0     | 2      | 0          | 28    |


#### shallow_sgd.yaml
epochs: 50  
batch_size: 32  
optimizer: sgd  
learning_rate: 0.01  
image_size: 128  
num_blocks: 2  
filters_start: 32  
dropout_rate: 0.3  
dense_units: 256  

Total params: 16,798,533 (64.08 MB)  
Trainable params: 16,798,341 (64.08 MB)  
Non-trainable params: 192 (768.00 B)  

Test accuracy : 0.7575  
Test loss     : 0.7070

#### Test with custom test-images (data/test-images)
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Dandelion      | 0.48      | 0.47   | 0.47     | 30      |
| Daisy          | 0.96      | 0.77   | 0.85     | 30      |
| Tulips         | 0.48      | 0.40   | 0.44     | 30      |
| Sunflowers     | 0.53      | 0.87   | 0.66     | 30      |
| Roses          | 0.70      | 0.53   | 0.60     | 30      |
| **Accuracy**   |           |        | **0.61** | **150** |
| Macro Avg      | 0.63      | 0.61   | 0.60     | 150     |
| Weighted Avg   | 0.63      | 0.61   | 0.60     | 150     |

| True \ Predicted | Dandelion | Daisy | Tulips | Sunflowers | Roses |
|------------------|-----------|-------|--------|------------|-------|
| **Dandelion**    | 14        | 0     | 0      | 15         | 1     |
| **Daisy**        | 5         | 23    | 1      | 1          | 0     |
| **Tulips**       | 5         | 0     | 12     | 7          | 6     |
| **Sunflowers**   | 4         | 0     | 0      | 26         | 0     |
| **Roses**        | 1         | 1     | 12     | 0          | 16    |


#### wide_rmsprop.yaml
epochs: 50  
batch_size: 64  
optimizer: rmsprop  
learning_rate: 0.0005  
image_size: 128  
num_blocks: 4  
filters_start: 64  
dropout_rate: 0.5  
dense_units: 512  

Total params: 18,335,109 (69.94 MB)  
Trainable params: 18,333,189 (69.94 MB)  
Non-trainable params: 1,920 (7.50 KB)  

Test accuracy : 0.7575  
Test loss     : 0.6159

#### Test with custom test-images (data/test-images)
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Dandelion    | 0.64      | 0.23   | 0.34     | 30      |
| Daisy        | 0.82      | 0.93   | 0.88     | 30      |
| Tulips       | 0.74      | 0.47   | 0.57     | 30      |
| Sunflowers   | 0.53      | 1.00   | 0.69     | 30      |
| Roses        | 0.86      | 0.83   | 0.85     | 30      |
| **Accuracy** |           |        | **0.69** | **150** |
| Macro Avg    | 0.72      | 0.69   | 0.67     | 150     |
| Weighted Avg | 0.72      | 0.69   | 0.67     | 150     |

| True \ Predicted | Dandelion | Daisy | Tulips | Sunflowers | Roses |
|------------------|-----------|-------|--------|------------|-------|
| **Dandelion**    | 7         | 5     | 1      | 17         | 0     |
| **Daisy**        | 1         | 28    | 0      | 1          | 0     |
| **Tulips**       | 2         | 1     | 14     | 9          | 4     |
| **Sunflowers**   | 0         | 0     | 0      | 30         | 0     |
| **Roses**        | 1         | 0     | 4      | 0          | 25    |

#### alexnet.yaml

model: alexnet  
image_size: 224  
dropout_rate: 0.5  
optimizer: adam  
learning_rate: 0.0001  
batch_size: 32  
epochs: 50  

Total params: 58,303,237 (222.41 MB)
Trainable params: 58,302,533 (222.41 MB)
Non-trainable params: 704 (2.75 KB)

Test accuracy : 0.8283
Test loss     : 0.4361

#### Test with custom test-images (data/test-images)
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Dandelion    | 0.74      | 0.47   | 0.57     | 30      |
| Daisy        | 1.00      | 0.77   | 0.87     | 30      |
| Tulips       | 0.92      | 0.40   | 0.56     | 30      |
| Sunflowers   | 0.44      | 0.90   | 0.59     | 30      |
| Roses        | 0.85      | 0.93   | 0.89     | 30      |
| **Accuracy** |           |        | **0.69** | **150** |
| Macro Avg    | 0.79      | 0.69   | 0.69     | 150     |
| Weighted Avg | 0.79      | 0.69   | 0.69     | 150     |

| True \ Predicted | Dandelion | Daisy | Tulips | Sunflowers | Roses |
|------------------|-----------|-------|--------|------------|-------|
| **Dandelion**    | 14        | 0     | 0      | 16         | 0     |
| **Daisy**        | 0         | 23    | 0      | 7          | 0     |
| **Tulips**       | 1         | 0     | 12     | 12         | 5     |
| **Sunflowers**   | 3         | 0     | 0      | 27         | 0     |
| **Roses**        | 1         | 0     | 1      | 0          | 28    |

#### cnn_svm.yaml
backbone: baseline  
image_size: 128  
classifier: svm  
svm_C: 10  
svm_kernel: rbf

Val accuracy : 0.8147

#### Test with custom test-images (data/test-images)
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Dandelion    | 0.59      | 0.80   | 0.68     | 30      |
| Daisy        | 1.00      | 0.83   | 0.91     | 30      |
| Tulips       | 0.79      | 0.50   | 0.61     | 30      |
| Sunflowers   | 0.77      | 0.80   | 0.79     | 30      |
| Roses        | 0.82      | 0.93   | 0.88     | 30      |
| **Accuracy** |           |        | **0.77** | **150** |
| Macro Avg    | 0.79      | 0.77   | 0.77     | 150     |
| Weighted Avg | 0.79      | 0.77   | 0.77     | 150     |

| True \ Predicted | Dandelion | Daisy | Tulips | Sunflowers | Roses |
|------------------|-----------|-------|--------|------------|-------|
| **Dandelion**    | 24        | 0     | 2      | 4          | 0     |
| **Daisy**        | 3         | 25    | 0      | 2          | 0     |
| **Tulips**       | 8         | 0     | 15     | 1          | 6     |
| **Sunflowers**   | 6         | 0     | 0      | 24         | 0     |
| **Roses**        | 0         | 0     | 2      | 0          | 28    |