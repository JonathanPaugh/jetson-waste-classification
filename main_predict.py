from configs.model import DATASET_TEST_PATH
from core.loader import load_train_dataset, load_test_dataset
from core.model import compile_model, train_model
from library.option_input import run_menu, OptionInput
from utils.pickle import import_trained_model, has_trained_model
from utils_jetson.sensor_sniffer import sniff


def convert_prediction_to_class(prediction, class_names):
    prediction_list = list(prediction)
    max_probability = max(prediction_list)
    max_index = prediction_list.index(max_probability)
    return class_names[max_index]

def main():
    train_data, valid_data = load_train_dataset()
    model = compile_model(num_classes=len(train_data.class_names))

    def predict_test():
        train_model(model, train_data, valid_data)
        test_data = load_test_dataset()

        predictions = model.predict(test_data)
        prediction_classes = [convert_prediction_to_class(prediction, train_data.class_names) for prediction in
                              predictions]

        print(prediction_classes)

    def predict_jetson():
        if not has_trained_model():
            print("Jetson nano requires an exported trained model")
            return

        import_trained_model(model)

        print(sniff())

    run_menu("Prediction Menu", [
        OptionInput.MENU_EXIT,
        (f'Predict from {DATASET_TEST_PATH}', predict_test),
        (f'Predict jetson nano', predict_jetson),
    ])


if __name__ == '__main__':
    main()
