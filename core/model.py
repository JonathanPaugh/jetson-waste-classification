from tensorflow.python.keras.layers import Dropout, Dense, Input, \
    RandomFlip, RandomRotation, RandomZoom
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping

from core.transfer_learning import ApplicationBasedTransferLearningModel
from utils.pickle import has_trained_model, import_trained_model, export_trained_model
import configs.model as config
import configs.transfer_learning as config_tl


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


def compile_model(num_classes):
    image_size = config_tl.TRANSFER_LEARNING_MODEL.IMAGE_SIZE or config.IMAGE_SIZE
    INPUT_SHAPE = (*image_size, 3)  # 3 channels for RGB

    data_augmentation = Sequential([
        RandomFlip('horizontal_and_vertical'),
        RandomRotation(0.2),
        RandomZoom(0.1),
    ], name='augmentation')

    base_model = config_tl.TRANSFER_LEARNING_MODEL(
        input_shape=INPUT_SHAPE,
        trainable=False,
    )

    inputs = Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(config.MODEL_DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.summary()
    model.compile(
        optimizer='adam',
        **_build_model_compile_params(),
    )
    return model


def recompile_model_for_fine_tuning(model, unfreeze_breakpoint, learning_rate):
    """
    Recompiles the given model for fine tuning.
    :param model: the base model (feature extractor)
    :param unfreeze_breakpoint: the layer above which all layers should be unfrozen
    :param learning_rate: the rate at which the unfrozen layers learn
    :return: whether the model can be fine-tuned
    """
    base_model = config_tl.TRANSFER_LEARNING_MODEL.find_base_model(model)
    if base_model is None:
        print('WARNING: Failed to locate base model; fine-tuning will be skipped', model.layers)
        return False

    base_model.unfreeze(unfreeze_breakpoint)

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

    if not config_tl.FINE_TUNING_ENABLED:
        return _history.history

    unfreeze_breakpoints = config_tl.FINE_TUNING_UNFREEZE_BREAKPOINTS \
        if issubclass(config_tl.TRANSFER_LEARNING_MODEL, ApplicationBasedTransferLearningModel) \
        else (None,)  # layer-based models are incompatible with breakpoints
    for i, unfreeze_breakpoint in enumerate(unfreeze_breakpoints):
        learning_rate = config_tl.FINE_TUNING_LEARNING_RATE \
            * (1 / config_tl.FINE_TUNING_LEARNING_RATE_DECAY) ** -i
        if not recompile_model_for_fine_tuning(model,
            unfreeze_breakpoint=unfreeze_breakpoint,
            learning_rate=learning_rate): break

        print(f'Fine tuning model for up to'
            f' {config_tl.FINE_TUNING_NUM_EPOCHS} additional epochs...')

        _history_fine = model.fit(
            train_data,
            epochs=last_epoch + config_tl.FINE_TUNING_NUM_EPOCHS,
            initial_epoch=last_epoch,
            **fit_params,
        )
        _merge_histories(_history, _history_fine)
        if use_export:
            export_trained_model(model, _history.history)

        last_epoch = _history_fine.epoch[-1] + 1

    return _history.history
