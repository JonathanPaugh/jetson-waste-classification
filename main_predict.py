from configs.model import DATASET_TEST_PATH
from core.loader import load_train_dataset, load_test_dataset, load_image_tensor
from core.model import compile_model, train_model
from library.option_input import run_menu, OptionInput
from utils.pickle import import_trained_model, has_trained_model
from utils_jetson.sensor_sniffer import sniff
import tensorflow

def predict_classes(model, test_data, class_names):
    return [
        convert_prediction_to_class(prediction, class_names)
        for prediction in model.predict(test_data)
    ]

def convert_prediction_to_class(prediction, class_names):
    prediction_list = list(prediction)
    max_probability = max(prediction_list)
    max_index = prediction_list.index(max_probability)
    return class_names[max_index]

def main():
    train_data, valid_data = load_train_dataset()
    class_names = train_data.class_names
    model = compile_model(num_classes=len(class_names))

    def predict_test():
        train_model(model, train_data, valid_data)
        test_data = load_test_dataset()

        print(list(test_data))

        predictions = predict_classes(model, test_data, class_names)

        print(predictions)

    def predict_jetson():
        if not has_trained_model():
            print("Jetson nano requires an exported trained model")
            return

        import_trained_model(model)

        image = load_image_tensor('dataset/test/cardboard1.jpg')

        test_data = tensorflow.convert_to_tensor([image])

        predictions = predict_classes(model, test_data, class_names)

        print(predictions)
        print(sniff())

    run_menu("Prediction Menu", [
        OptionInput.MENU_EXIT,
        (f'Predict with {DATASET_TEST_PATH}', predict_test),
        (f'Predict jetson nano pipeline', predict_jetson),
    ])


if __name__ == '__main__':
    main()
