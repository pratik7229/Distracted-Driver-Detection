import tensorflow as tf
from config import *


def loadDataSet():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    return train_ds, validation_ds, class_names

def preprocessingDataset(train_ds, val_ds):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess_train(x, y):
        x = tf.image.resize_with_pad(x, IMG_SIZE[0], IMG_SIZE[1])
        x = normalization_layer(x)
        return x, y
    
    def preprocess_val(x, y):
        x = tf.image.resize_with_pad(x, IMG_SIZE[0], IMG_SIZE[1])
        x = normalization_layer(x)
        return x, y
    
    train_ds = train_ds.map(
        preprocess_train,
        num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_val,
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds