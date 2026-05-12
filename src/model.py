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
