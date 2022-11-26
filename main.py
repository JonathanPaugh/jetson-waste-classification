import matplotlib.pyplot as plt

from core.loader import load_dataset
from core.model import compile_model
from utils.plot import save_plot
from utils.pickle import has_saved_weights, load_weights, serialize_weights

import configs.model as config


def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='train_accuracy')
    ax.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_plot(fig, 'history.png')


def evaluate_model(model, test_data, verbose=2):
    _, test_acc = model.evaluate(test_data, verbose=verbose)
    print(f'Accuracy on test set after {config.MODEL_NUM_EPOCHS} epoch(s):'
          + f' {test_acc * 100:.2f}%')

def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='accuracy')
    ax.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.ylim([0.5, 1])
    ax.legend()

    try:
        mkdir(config.OUTPUT_PATH)
    except FileExistsError:
        pass

    fig_path = join(config.OUTPUT_PATH, 'history.png')
    fig.savefig(fig_path, bbox_inches='tight')

def evaluate_model(model, test_data, verbose=2):
    test_images, test_labels = test_data
    _, test_acc = model.evaluate(test_images, test_labels, verbose=verbose)
    print(f'Accuracy on test set: {test_acc*100:.2f}%')

def main():
    train_data, test_data = load_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    if has_saved_weights():
        load_status = load_weights(model)
    else:
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=config.MODEL_NUM_EPOCHS,
            batch_size=config.MODEL_BATCH_SIZE,
            workers=config.MODEL_WORKERS,
            use_multiprocessing=True
        )
        serialize_weights(model)
        plot_history(history)
    evaluate_model(model, test_data)


if __name__ == '__main__':
    main()
