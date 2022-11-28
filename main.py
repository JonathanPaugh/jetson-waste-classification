import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from core.loader import load_dataset
from core.model import compile_model
from utils.plot import save_plot



import configs.model as config
from core.loader import load_train_dataset
from core.model import compile_model, train_model
from utils.plot import save_plot
from utils_jetson.hardware import tweak_hardware_settings


def plot_accuracy(history):
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Training accuracy')
    ax.plot(history['val_accuracy'], label='Validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_plot(fig, 'accuracy.png')

def plot_f1_score(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['f1_m'], label='train_f1_score')
    ax.plot(history.history['val_f1_m'], label='val_f1_score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('f1_score')
    ax.legend()
    save_plot(fig, 'f1.png')

def plot_confusion_matrix(model, test_data):
    y_actual = []
    y_pred = []
    for x, y in test_data:
        y_actual.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x), axis=-1))
    cm = confusion_matrix(y_actual, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(test_data.class_names)))
    ax.set_yticks(np.arange(len(test_data.class_names)))
    ax.set_xticklabels(test_data.class_names)
    ax.set_yticklabels(test_data.class_names)
    ax.set_title('Confusion Matrix')
    for i in range(len(test_data.class_names)):
        for j in range(len(test_data.class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='blue')
    save_plot(fig, 'confusion_matrix.png')



def evaluate_model(model, test_data, verbose=2):
    loss, accuracy, f1_score, precision, recall  = model.evaluate(test_data, verbose=verbose)
    print(f"After epoch {config.MODEL_NUM_EPOCHS} the model has a loss of {loss:.2f}, \
          an accuracy of {accuracy:.2f}, a f1 score of {f1_score:.2f},\
          a precision of {precision:.2f}, and a recall of {recall:.2f}.")
 
def main():
    tweak_hardware_settings()

    train_data, test_data = load_train_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    history = train_model(model, train_data, test_data)
    plot_accuracy(history)
    evaluate_model(model, test_data)
    plot_confusion_matrix(model, test_data)


if __name__ == '__main__':
    main()

