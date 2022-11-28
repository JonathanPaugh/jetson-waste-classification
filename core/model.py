from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    Input, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow_hub import KerasLayer, load as load_model
from utils.pickle import has_trained_model, import_trained_model, export_trained_model
import configs.model as config


def _build_model_fit_params(**kwargs):
    return dict(
        batch_size=config.MODEL_BATCH_SIZE,
        workers=config.MODEL_WORKERS,
        use_multiprocessing=True,
        callbacks=[EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=config.MODEL_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )],
        **kwargs,
    )

def _build_model_compile_params(**kwargs):
    return dict(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        **kwargs,
    )

def _merge_histories(a, b):
    for metric in a.history:
        a.history[metric] += b.history[metric]


def compile_model(num_classes):
    INPUT_SHAPE = (*config.IMAGE_SIZE, 3)  # 3 channels for RGB

    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.1),
    ])

    feature_extractor = KerasLayer(
        config.MODEL_FEATURE_EXTRACTOR,
        input_shape=INPUT_SHAPE,
        trainable=False,
    )

    # HACK: define model via functional API to ensure feature extractor runs in inference mode
    inputs = Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)
    x = Rescaling(1./255)(x)
    x = feature_extractor(x, training=False)  # force run in inference mode
    x = Dropout(config.MODEL_DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.summary()
    model.compile(
        optimizer='adam',
        **_build_model_compile_params(),
    )
    return model


def recompile_model_for_fine_tuning(model):
    feature_extractor = next((l for l in model.layers
        if isinstance(l, KerasLayer)), None)
    if feature_extractor is None:
        return False  # fail silently if no feature extractor found

    feature_extractor.trainable = True
    feature_extractor.arguments = dict(
        batch_norm_momentum=config.MODEL_FINE_TUNING_BATCH_NORM_MOMENTUM
    )

    model.summary()
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=config.MODEL_FINE_TUNING_LEARNING_RATE),
        **_build_model_compile_params(),
    )
    return True


def train_model(model, train_data, test_data, use_import=True, use_export=True):
    if use_import and has_trained_model():
        return import_trained_model(model)

    fit_params = _build_model_fit_params(validation_data=test_data)

    print(f'Fitting model for up to'
        f' {config.MODEL_NUM_EPOCHS} epochs...')
    _history = model.fit(
        train_data,
        epochs=config.MODEL_NUM_EPOCHS,
        **fit_params
    )
    last_epoch = _history.epoch[-1] + 1
    print(f'Model fitting complete in {last_epoch} epoch(s)')
    if use_export:
        export_trained_model(model, _history.history)

    if config.MODEL_FINE_TUNING and recompile_model_for_fine_tuning(model):
        print(f'Fine tuning model for up to'
            f' {config.MODEL_FINE_TUNING_NUM_EPOCHS} additional epochs...')
        _history_fine = model.fit(
            train_data,
            epochs=last_epoch + config.MODEL_FINE_TUNING_NUM_EPOCHS,
            initial_epoch=last_epoch,
            **fit_params,
        )
        print(f'Model fine tuning complete in {_history_fine.epoch[-1] + 1} epoch(s)')
        _merge_histories(_history, _history_fine)  # fit creates two separate history dicts
        if use_export:
            export_trained_model(model, _history.history)

    return _history.history
