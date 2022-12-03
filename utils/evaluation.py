import matplotlib.pyplot as plt
import numpy as np
import configs.model as config

from utils.plot import save_plot

if not config.IS_JETSON:
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

def evaluate_model(model, test_data, verbose=2):
    _, test_acc = model.evaluate(test_data, verbose=verbose)
    print(f'Accuracy on test set:'
          f' {test_acc * 100:.2f}%')

def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Training accuracy')
    ax.plot(history['val_accuracy'], label='Validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_plot(fig, 'history.png')
    plt.close(fig)

def plot_confusion_matrix(y_actual, y_pred, test_data):
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
            if (cm[i, j] >  cm.max() / 2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
            else:   
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    

            
    fig.colorbar(ax.imshow(cm, cmap=plt.cm.Blues))
    save_plot(fig, 'confusion_matrix.png')
    plt.close(fig)

def get_predicted_vs_actual(model, test_data):
    y_actual = []
    y_pred = []
    for x, y in test_data:
        y_actual.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x), axis=-1))
    return y_actual, y_pred

def output_training_results(history, model, test_data):
    evaluate_model(model, test_data)

    # Anything that requires display below here
    if config.IS_JETSON:
        return

    plot_history(history)

    y_actual, y_pred = get_predicted_vs_actual(model, test_data)
    plot_confusion_matrix(y_actual, y_pred, test_data)

    print('Classification Report')
    print(classification_report(y_actual, y_pred, target_names=test_data.class_names))
    with open('dist/classification_report.txt', 'a') as f:
        f.write(classification_report(y_actual, y_pred, target_names=test_data.class_names))
      
   