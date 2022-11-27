from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    InputLayer, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping
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


def recompile_model_for_fine_tuning(model):
    feature_extractor = next((l for l in model.layers
        if isinstance(l, KerasLayer)), None)
    if feature_extractor is None:
        return False  # fail silently if no feature extractor found

    feature_extractor.trainable = True
    model.summary()
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return True


def _build_fit_params(**kwargs):
    return {
        'batch_size': config.MODEL_BATCH_SIZE,
        'workers': config.MODEL_WORKERS,
        'use_multiprocessing': True,
        **kwargs,
    }


def train_model(model, train_data, test_data, use_import=True, use_export=True):
    if use_import and has_trained_model():
        return import_trained_model(model)

    fit_params = _build_fit_params(validation_data=test_data)
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=config.MODEL_EARLY_STOPPING_PATIENCE,
    )

    _history = model.fit(
        train_data,
        epochs=config.MODEL_NUM_EPOCHS,
        callbacks=[early_stopping],
        **fit_params
    )
    if use_export:
        export_trained_model(model, _history.history)

    if config.MODEL_FINE_TUNING and recompile_model_for_fine_tuning(model):
        _history = model.fit(
            train_data,
            epochs=_history.epoch[-1] + config.MODEL_NUM_FINE_TUNING_EPOCHS,
            initial_epoch=_history.epoch[-1],
            callbacks=[early_stopping],
            **fit_params,
        )
        if use_export:
            export_trained_model(model, _history.history)

    return _history.history
