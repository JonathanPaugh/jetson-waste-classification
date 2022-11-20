from core.loader import load_dataset


def main():
    train_data, val_data = load_dataset()
    print(train_data.class_names)


if __name__ == '__main__':
    main()
