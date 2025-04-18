# Imports
import tensorflow as tf

# Dataset assembly
def dataset_assembly(raw_data_path, subfolders, batch_size, img_height=180, img_width=180, seed=123):
    """
    Assembles a dataset from a folder structure.
    Parameters:
    - raw_data_path: Path to the raw data folder.
    - subfolders: List of subfolders to include in the dataset.
    - batch_size: Size of the batches of data.
    - img_height: Height of the images.
    - img_width: Width of the images.
    - seed: Random seed for shuffling.
    Returns:
    - dataset: A TensorFlow dataset object.
    """
    print(f"\n--Assembling dataset--")
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            raw_data_path,
            labels="inferred",
            label_mode="int",
            class_names=subfolders,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=(img_height, img_width),
            shuffle=True,
            seed=seed,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False
        )

        class_names = dataset.class_names
        print(f"Detected classes: {class_names}")

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        for images, labels in dataset.take(1):
            print(f"Images batch shape : {images.shape}")

        print(f"Dataset assembly completed.")
        return dataset, class_names

    except Exception as e:
        print(f"Error creating dataset: {e}")


# Dataset split
def dataset_split(dataset, train_split=0.8, val_split=0.1, batch_size=1000):
    """
    Splits a dataset into training, validation, and test sets.
    Parameters:
    - dataset: The dataset to split.
    - train_split: Proportion of the dataset to use for training.
    - val_split: Proportion of the dataset to use for validation.
    - test_split: Proportion of the dataset to use for testing.
    Returns:
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - test_ds: Test dataset.
    """
    print(f"\n--Making dataset split--")
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()  # Faster than len(dataset)
    print(f"Dataset size: {dataset_size*batch_size}")
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    print(f"Creating training set of size ~{train_size*batch_size}...")
    train_ds = dataset.take(train_size)
    remaining_ds = dataset.skip(train_size)
    print(f"Creating validation set of size ~{val_size*batch_size}...")
    val_ds = remaining_ds.take(val_size)
    print(f"Creating test set of size ~{test_size*batch_size}...")
    test_ds = remaining_ds.skip(val_size)

    print(f"Dataset split completed.")
    return train_ds, val_ds, test_ds