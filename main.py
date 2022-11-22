import matplotlib.pyplot as plt

from core.loader import load_dataset
from core.model import compile_model
from utils.plot import save_plot
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
    print(f'Accuracy on test set after {config.NUM_EPOCHS} epoch(s):'
        + f' {test_acc*100:.2f}%')

def main():
    train_data, test_data = load_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=config.NUM_EPOCHS,
    )
    plot_history(history)
    evaluate_model(model, test_data)


if __name__ == '__main__':
    main()
