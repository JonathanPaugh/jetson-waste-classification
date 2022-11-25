import math
from core.loader import load_train_dataset, load_test_dataset
from core.model import compile_model, train_model

def convert_prediction_to_class(prediction, class_names):
    prediction_list = list(prediction)
    max_probability = max(prediction_list)
    max_index = prediction_list.index(max_probability)
    return class_names[max_index]

def main():
    train_data, test_data = load_train_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    train_model(model, train_data, test_data)

    test_data = load_test_dataset()

    predictions = model.predict(test_data)
    prediction_classes = [convert_prediction_to_class(prediction, train_data.class_names) for prediction in predictions]

    print(prediction_classes)

if __name__ == '__main__':
    main()
