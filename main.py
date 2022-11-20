from os import mkdir
from os.path import join

from keras import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

from core.loader import load_dataset
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
    model.summary()

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=config.NUM_EPOCHS,
    )
    evaluate_model(model, test_data)
    plot_history(history)


if __name__ == '__main__':
    main()
