from keras import Input, layers, models

def build_cnn(input_shape, num_classes, num_blocks=4, filters_start=32, dropout_rate=0.5, dense_units=512):
    model_layers = [
        Input(shape=input_shape),

        # augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2, value_range=(0, 1)),
        layers.RandomContrast(0.2),
    ]

    for i in range(num_blocks):
        filters = filters_start * (2 ** i)
        model_layers += [
            layers.Conv2D(filters, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
        ]

    model_layers += [
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ]

    return models.Sequential(model_layers)


def build_alexnet(input_shape, num_classes, dropout_rate=0.5):
    model_layers = [
        Input(shape=input_shape),

        # augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2, value_range=(0, 1)),
        layers.RandomContrast(0.2),

        # Block 1 – large 11×11 kernel, stride 4 (original AlexNet)
        layers.Conv2D(96, (11, 11), strides=4, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),

        # Block 2 – 5×5 kernel
        layers.Conv2D(256, (5, 5), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),

        # Blocks 3–5 – 3×3 kernels, no pooling between 3 and 4
        layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(3, strides=2),

        # Classifier – two 4096-unit dense layers
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ]

    return models.Sequential(model_layers)