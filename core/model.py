from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    Input, RandomFlip, RandomRotation, RandomZoom
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from utils.pickle import has_trained_model, import_trained_model, export_trained_model
import configs.model as config


def _index_layer(layers, name):
    return layers.index(next((l for l in layers
        if l.name == name)))

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
            verbose=1,
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


BASE_MODEL_FACTORY = InceptionV3
BASE_MODEL_NAME = 'inception_v3'
BASE_MODEL_FREEZE_BREAKPOINTS = [_index_layer(base_model.layers, layer_name)
    for layer_name in ('mixed9', 'mixed8', 'mixed7', 'mixed6', 'mixed5')]


def compile_model(num_classes):
    INPUT_SHAPE = (*config.IMAGE_SIZE, 3)  # 3 channels for RGB

    data_augmentation = Sequential([
        RandomFlip('horizontal_and_vertical'),
        RandomRotation(0.2),
        RandomZoom(0.1),
    ], name='augmentation')

    base_model = BASE_MODEL_FACTORY(input_shape=INPUT_SHAPE, include_top=False)
    base_model.trainable = False
    base_model.summary()

    inputs = Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = Flatten()(x)
    x = Dropout(config.MODEL_DROPOUT_RATE)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(config.MODEL_DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.summary()
    model.compile(
        optimizer='adam',
        **_build_model_compile_params(),
    )
    return model


def recompile_model_for_fine_tuning(model, freeze_breakpoint, learning_rate):
    """
    Recompiles the given model for fine tuning.
    :param model: the base model (feature extractor)
    :param freeze_breakpoint: the layer above which all layers should be unfrozen
    :param learning_rate: the rate at which the unfrozen layers learn
    :return: whether the model can be fine-tuned
    """
    try:
        base_model = next((l for l in model.layers
            if l.name == BASE_MODEL_NAME))
    except StopIteration:
        return False  # fail silently if no feature extractor found

    base_model.trainable = True
    for layer in base_model.layers[:freeze_breakpoint+1]:
        layer.trainable = False

    model.summary()
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=learning_rate),
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
    if use_export:
        export_trained_model(model, _history.history)

    if not config.MODEL_FINE_TUNING:
        return _history.history

    for i, freeze_breakpoint in enumerate(BASE_MODEL_FREEZE_BREAKPOINTS):
        recompile_model_for_fine_tuning(model,
            freeze_breakpoint=freeze_breakpoint,
            learning_rate=(config.MODEL_FINE_TUNING_LEARNING_RATE * 10 ** -i))
        print(f'Fine tuning model for up to'
            f' {config.MODEL_FINE_TUNING_NUM_EPOCHS} additional epochs...')
        _history_fine = model.fit(
            train_data,
            epochs=last_epoch + config.MODEL_FINE_TUNING_NUM_EPOCHS,
            initial_epoch=last_epoch,
            **fit_params,
        )
        _merge_histories(_history, _history_fine)
        if use_export:
            export_trained_model(model, _history.history)
        last_epoch = _history_fine.epoch[-1] + 1

    return _history.history
