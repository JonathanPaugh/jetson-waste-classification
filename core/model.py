from tensorflow.python.keras.layers import Rescaling, MaxPooling2D, Flatten, Dense, Conv2D
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import Sequential

from utils.pickle import has_trained_model, import_trained_model, export_trained_model
import configs.model as config


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


def train_model(model, train_data, test_data, use_import=True, use_export=True):
    if use_import and has_trained_model():
        history = import_trained_model(model)
    else:
        _history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=config.MODEL_NUM_EPOCHS,
            batch_size=config.MODEL_BATCH_SIZE,
            workers=config.MODEL_WORKERS,
            use_multiprocessing=True
        )

        history = _history.history
        if use_export:
            export_trained_model(model, history)

    return history

