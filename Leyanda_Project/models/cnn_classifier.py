# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image

# Model creation
def create_model(model_name, input_shape=(180, 180, 3), class_names=None, train_transfer_model=False, transfer_learning=False, target_binary_class_name=None):
    """
    Creates an image classification model with customizable architecture.
    Parameters:
    - model_name: Name of the model
    - input_shape: Shape of the input images
    - class_names: List of class names for the dataset
    - train_transfer_model: If True, the transfer learning model will be trained
    - transfer_learning: If True, a transfer learning model will be created
    - target_binary_class_name: Name of the target binary class for binary classification, if None, a multi-class classification model will be created
    Returns:
    - Model ready for training
    """
    print(f"\n--Creating model: {model_name}--")

    num_classes=len(class_names)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(factor=0.2)
    ])

    inputs = tf.keras.Input(shape=input_shape)

    x = data_augmentation(inputs, training=True)

    if transfer_learning:
        base_model = VGG16(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = train_transfer_model

        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)


    else:
        x = tf.keras.layers.Rescaling(1./255)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

    if target_binary_class_name:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name=model_name)

    return model

# Model compilation
def check_and_compile_model(model, model_name, target_binary_class_name):
    """
    Check if the model is compiled, compile it with default settings if not.
    Parameters:
    - model: The model object to check and compile
    - model_name: Name of the model (for logging)
    - target_binary_class_name: Name of the target binary class for binary classification
    Returns:
    - The compiled model
    """
    print(f"\n--Compiling {model_name}--")
    if target_binary_class_name:
        loss = "binary_crossentropy"
        print(f"Using binary classification loss: {loss}")
    else:
        loss = "sparse_categorical_crossentropy"
        print(f"Using classification loss: {loss}")

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    return model

# Model training
def train_model(model, model_name, train_ds, val_ds, epochs=10, save_path=None, class_weight=False, target_binary_class_name=None, callbacks=None, wandb=None):
    """
    Train a single model and optionally save it.
    Parameters:
    - model: The model object to train
    - model_name: Name of the model (for logging)
    - train_ds: Training dataset
    - val_ds: Validation dataset
    - epochs: Number of epochs to train
    - save_path: Path to save the model
    - class_weight: If True, class weights will be used for imbalanced datasets
    - target_binary_class_name: Name of the target binary class for binary classification
    - callbacks: List of callbacks to use during training
    - wandb: Wandb object for logging
    Returns:
    - The trained model
    """
    print(f"\n--Training {model_name}--")

    model = check_and_compile_model(model, model_name, target_binary_class_name)
    if class_weight:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            class_weight = add_class_weights()
        )

    else :
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, f"{model_name}.keras")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        artifact = wandb.Artifact(name=f"{model_name}_model", type="model")
        artifact.add_file(model_save_path)
        if wandb:
            wandb.log_artifact(artifact)
    return model

# Class weights
def add_class_weights(train_ds):
    """
    Compute class weights for imbalanced datasets.
    Parameters:
    - train_ds: Training dataset
    Returns:
    - class_weights_dict: Dictionary mapping class indices to weights
    """
    print("\n--Computing class weights--")
    y_train = []

    for _, labels in train_ds:
        y_train.extend(labels.numpy())

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights :", class_weights_dict)
    return class_weights_dict
