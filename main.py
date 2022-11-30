from core.loader import load_train_dataset
from core.model import compile_model, train_model
from utils_jetson.hardware import tweak_hardware_settings
from utils.evaluation import get_evaluation_metrics


def main():
    tweak_hardware_settings()

    train_data, test_data = load_train_dataset()
    model = compile_model(num_classes=len(train_data.class_names))
    history = train_model(model, train_data, test_data)

    get_evaluation_metrics(history, model, test_data)


if __name__ == '__main__':
    main()

