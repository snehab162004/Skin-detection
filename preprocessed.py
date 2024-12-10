import tensorflow as tf

def load_data(data_dir, img_size=128):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=32,
        label_mode='categorical'  # For multi-class classification
    )

train_ds = load_data("./processed_dataset/train")
val_ds = load_data("./processed_dataset/val")
test_ds = load_data("./processed_dataset/test")
