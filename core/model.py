from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    InputLayer, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.python.keras.models import Sequential
from tensorflow_hub import KerasLayer
from utils.pickle import has_trained_model, import_trained_model, export_trained_model
import configs.model as config


def compile_model(num_classes):
    INPUT_SHAPE = (*config.IMAGE_SIZE, 3)  # 3 for RGB

    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.1),
    ])

    feature_extractor = KerasLayer(
        config.MODEL_FEATURE_EXTRACTOR,
        input_shape=INPUT_SHAPE,
        trainable=False
    )

    model = Sequential([
        InputLayer(input_shape=INPUT_SHAPE),
        Rescaling(1./255),
        data_augmentation,
        feature_extractor,
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ])
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
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
