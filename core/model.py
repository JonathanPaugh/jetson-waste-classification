from tensorflow.python.keras.layers import Rescaling, MaxPooling2D, Flatten, Dense, Conv2D
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import Sequential


def compile_model(num_classes):
    model = Sequential([
        Rescaling(1./255),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
    ])
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model
