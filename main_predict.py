from tensorflow.python.framework.ops import convert_to_tensor

from configs.model import DATASET_TEST_PATH
from core.loader import load_train_dataset, load_test_dataset, load_image_tensor
from core.model import compile_model, train_model
from library.option_input import run_menu, OptionInput
from utils.pickle import import_trained_model, has_trained_model
from utils_jetson import sensor_camera
from utils_jetson.hardware import tweak_hardware_settings
from utils_jetson.sensor_sniffer import sniff

def output_predictions(model, test_data, class_names):
    predictions = predict_classes(model, test_data, class_names)
    print(predictions)
    return predictions

def predict_classes(model, test_data, class_names):
    return [
        extract_best_prediction_class(prediction, class_names)
        for prediction in model.predict(test_data)
    ]

def extract_best_prediction_class(prediction, class_names):
    prediction_list = list(prediction)
    max_probability = max(prediction_list)
    max_index = prediction_list.index(max_probability)
    return class_names[max_index], prediction_list[max_index]

def main():
    tweak_hardware_settings()

    train_data, valid_data = load_train_dataset()
    class_names = train_data.class_names
    model = compile_model(num_classes=len(class_names))

    def predict_test():
        train_model(model, train_data, valid_data)
        test_data = load_test_dataset()

        output_predictions(model, test_data, class_names)

    def predict_jetson():
        if not has_trained_model():
            print("Jetson nano requires an exported trained model")
            return

        import_trained_model(model)

        def predict():
            print("Taking snapshot...")
            image = sensor_camera.snapshot()

            print("Predicting...")
            test_data = convert_to_tensor([image])
            output_predictions(model, test_data, class_names)

        run_menu("Ready to predict", [
            OptionInput.MENU_EXIT,
            (f'Predict snapshot', predict),
        ])

    def predict_scent():
        print(sniff())

    run_menu("Prediction Menu", [
        OptionInput.MENU_EXIT,
        (f'Predict from {DATASET_TEST_PATH} directory', predict_test),
        (f'Predict jetson nano pipeline', predict_jetson),
        (f'Predict scent', predict_scent),
    ])


if __name__ == '__main__':
    main()
