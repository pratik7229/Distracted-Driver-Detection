import tensorflow as tf
from data_loader import loadDataSet, preprocessingDataset
from model import build_model, compile_model
from utils import print_class_info
from config import EPOCHS


def main():
    # load data
    train_ds, val_ds, class_name = loadDataSet()

    # prerpocess dataset
    train_ds, val_ds = preprocessingDataset(train_ds, val_ds)

    # print calls info 
    print_class_info(class_names=class_name)

    # build model
    model = build_model(num_classes=len(class_name))

    # compile model
    model = compile_model(model)

    # train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # save model
    model.save("model.h5")

if __name__ == "__main__":
    main()