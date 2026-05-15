# Experiment Configs

Each YAML file defines one training run. Pass the path to `src/train.py`:

```bash
python src/train.py configs/baseline.yaml
```

Outputs are saved to `models/` prefixed with the config name (e.g. `baseline_best_model.keras`, `baseline_history.json`).

---

## `baseline.yaml`
4-block CNN with Adam. The reference configuration — balanced depth and width, standard hyperparameters. All other experiments are compared against this.

| Parameter | Value |
|---|---|
| Blocks | 4 (32→64→128→256 filters) |
| Dense | 512 |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.001 |
| Batch size | 32 |

---

## `shallow_sgd.yaml`
2-block CNN with SGD. A simpler, lower-capacity model that serves as a near-baseline. Fewer parameters means faster training but higher risk of underfitting. SGD with a higher learning rate tests whether a classic optimizer can compete with Adam on this task.

| Parameter | Value |
|---|---|
| Blocks | 2 (32→64 filters) |
| Dense | 256 |
| Dropout | 0.3 |
| Optimizer | SGD, lr=0.01 |
| Batch size | 32 |

---

## `wide_rmsprop.yaml`
4-block CNN with doubled filter counts and RMSprop. Higher capacity than the baseline (~4× more parameters) at the cost of longer training and higher overfitting risk. RMSprop is often a strong choice for image tasks and provides a third optimizer data point alongside Adam and SGD.

| Parameter | Value |
|---|---|
| Blocks | 4 (64→128→256→512 filters) |
| Dense | 512 |
| Dropout | 0.5 |
| Optimizer | RMSprop, lr=0.0005 |
| Batch size | 64 |

---

## `alexnet.yaml`
AlexNet-inspired architecture with Adam. Uses a larger input size (224×224) to match the original AlexNet design. Lower learning rate compared to the baseline to stabilize training on the deeper, more parameter-heavy architecture.

| Parameter | Value |
|---|---|
| Model | AlexNet |
| Image size | 224×224 |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.0001 |
| Batch size | 32 |
