import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils.plot import save_plot



def plot_confusion_matrix(y_actual, y_pred, test_data):
    cm = confusion_matrix(y_actual, y_pred)
    print("-------------------")
    print(classification_report(y_actual, y_pred))
    print("-------------------")
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
    plt.close(fig)

def get_predicted_vs_actual(model, test_data):
    y_actual = []
    y_pred = []
    for x, y in test_data:
        y_actual.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x), axis=-1))
    return y_actual, y_pred