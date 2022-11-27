import matplotlib.pyplot as plt
import configs.model as config
from core.loader import load_train_dataset
from core.model import compile_model, train_model
from utils.plot import save_plot
from utils_jetson.hardware import tweak_hardware_settings


def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='train_accuracy')
    ax.plot(history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_plot(fig, 'history.png')


def evaluate_model(model, test_data, verbose=2):
    _, test_acc = model.evaluate(test_data, verbose=verbose)
    print(f'Accuracy on test set:'
          f' {test_acc * 100:.2f}%')

def main():
    tweak_hardware_settings()

    train_data, test_data = load_train_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    history = train_model(model, train_data, test_data)
    plot_history(history)
    evaluate_model(model, test_data)


if __name__ == '__main__':
    main()
